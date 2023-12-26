import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import onnxruntime


#Local
from .generation_utils import softmax, log_softmax, sample_top_p


#Constants
ENGLISH_ID = 50259
ARABIC_ID = 50272


@dataclass
class Hypothesis:
    """
    Represents a hypothesis in sequence generation.

    Attributes:
        tokens (List[int]): List of tokens in the hypothesis.
        k_caches (np.ndarray): key caches for inference.
        v_caches (np.ndarray): value caches for inference.
        logprob (float): Log probability of the hypothesis.
        is_done (bool): Indicates whether the hypothesis is complete.
    """
    tokens: List[int]
    k_caches: Optional[np.ndarray] = None
    v_caches: Optional[np.ndarray] = None
    logprob: float = 0.0
    is_done: bool = False
 


class Inference:
    """
    Class for handling sequence generation inference.

    Attributes:
        encoder: ONNX runtime inference session for the encoder.
        decoder: ONNX runtime inference session for the decoder.
        _mode: Language mode ("English" or "Arabic").
    """
    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        mode: Optional[str] = "English"
    ):
        """
        Initializes the Inference instance.

        Parameters:
            encoder_path: Path to the encoder model.
            decoder_path: Path to the decoder model.
            mode: Language mode ("English" or "Arabic").
        """
        options = onnxruntime.SessionOptions()
        providers = ["CPUExecutionProvider"]
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.encoder = onnxruntime.InferenceSession(
            encoder_path, sess_options=options, providers=providers)
        self.decoder = onnxruntime.InferenceSession(
            decoder_path, sess_options=options, providers=providers)
        self._mode = mode
        self.reset()
       
    def reset(self):
        self.cross_k_cache = None
        self.cross_v_cache = None

    def encode(self, feats):
        _, cross_k_cache, cross_v_cache = self.encoder.run(None, {"mel":feats})  
        self.cross_k_cache = cross_k_cache
        self.cross_v_cache = cross_v_cache

    def get_inits(self):
        lang_id = ENGLISH_ID if self._mode == "English" else ARABIC_ID
        k_caches = np.zeros((6, 1, 8, 1, 64)).astype(np.float32)
        v_caches = np.zeros((6, 1, 8, 1, 64)).astype(np.float32)
        tokens =  [50258, lang_id, 50359, 50363]
        hyp: Hypothesis = Hypothesis(
            tokens,
            k_caches,
            v_caches,
        )
        return hyp  

    def set_mode(self, mode: str):
        self._mode = mode   

    def __call__(
        self,
        hyp: Hypothesis,
        initial: Optional[bool] = False,
    ) -> Tuple[np.ndarray]:
        """
        Generates logits for the given hypothesis using the encoder and decoder.

        Parameters:
            hyp: The hypothesis.
            initial: Whether it's the initial generation or not.

        Returns:
            np.ndarray: Logits for the hypothesis.
            np.ndarray: keys caches for inference.
            np.ndarray: values caches for inference.
        """
        if initial:
            tokens = np.array(hyp.tokens)
        else:
            tokens = np.array([hyp.tokens[-1]])  
        tokens = np.expand_dims(tokens, axis=0).astype(np.int32)
        ort_inputs = {
            "tokens":tokens,
            "self_k_caches":hyp.k_caches,
            "self_v_caches":hyp.v_caches,
            "cross_k_caches":self.cross_k_cache,
            "cross_v_caches":self.cross_v_cache,
        }               
        outs = self.decoder.run(None, ort_inputs)  
        # update the internal variables
        # update the k an v states 
        k_caches = outs[1]  
        v_caches = outs[2]
        return (outs[0][:, -1: ,:]).squeeze(), k_caches, v_caches        


class Decoding:
    """
    Base class for decoding strategies.

    Attributes:
        inference: The inference instance.
    """

    def __init__(self, inference: Inference):
        """
        Initializes the Decoding instance.

        Parameters:
            inference: The inference instance.
        """
        self.inference = inference
        self.reset()

    def reset(self):
        self.inference.reset()

    def update(
        self,
        logits: np.ndarray,
        hyps: List[Hypothesis],
        initial: Optional[bool] = False
    ):
        """
        Updates hypotheses based on logits.

        Parameters:
            logits: The logits from the model.
            hyps: List of hypotheses to update.
        """
        pass

    def finalize(self, hyps: List[Hypothesis]):
        """
        Finalizes the decoding process.

        Parameters:
            hyps: List of hypotheses.
        """
        pass

    def set_mode(self, mode):
        self.inference.set_mode(mode)



class MaximumLikelihoodRanker:
    """
    Selects the sample with the highest log probabilities, penalized using either
    a simple length normalization or Google NMT paper's length penalty.
    """

    def __init__(self, length_penalty: Optional[float]=1.0):
        """
        Initializes the MaximumLikelihoodRanker instance.

        Parameters:
            length_penalty (Optional[float]): Length penalty factor.
        """
        self.length_penalty = length_penalty

    def rank(self, hyps: List[Hypothesis]):
        """
        Ranks hypotheses based on log probabilities and length penalties.

        Parameters:
            hyps: List of hypotheses.

        Returns:
            int: Index of the selected hypothesis.
        """
        def scores(logprobs, lengths):
            result = []
            for logprob, length in zip(logprobs, lengths):
                if self.length_penalty == 1.0:
                    penalty = length
                else:
                    # from the Google NMT paper
                    penalty = ((5.0 + length) / 6.0) ** self.length_penalty
                result.append(logprob / penalty)
            return result

        # get the sequence with the highest score
        lengths = [len(hyp.tokens) for hyp in hyps]
        sum_logprobs = [hyp.logprob for hyp in hyps]
        idx = int(np.argmax(scores(sum_logprobs, lengths)))
        return idx


class GreedyDecoding(Decoding):
    """
    Greedy decoding strategy for sequence generation.

    Attributes:
        inference: The inference instance.
        eos_id: End-of-sequence token ID.
        temperature: Temperature parameter for softmax.
        top_p: Top-p sampling parameter.
    """

    def __init__(
        self,
        inference,
        eos_id,
        temperature=0.9,
        top_p=0.95
    ):
        """
        Initializes the GreedyDecoding instance.

        Parameters:
            inference: The inference instance.
            eos_id: End-of-sequence token ID.
            temperature (float): Temperature parameter for softmax.
            top_p (float): Top-p sampling parameter.
        """
        super().__init__(inference)
        self.eos_id = eos_id
        self.temperature = temperature
        self.top_p = top_p

    def update(self, logits: np.ndarray, hyp: Hypothesis):
        """
        Updates a hypothesis based on logits using the greedy decoding strategy.

        Parameters:
            logits: The logits from the model.
            hyp: The hypothesis.

        Returns:
            Hypothesis: Updated hypothesis.
        """
        logits = logits.reshape(-1,)
        if self.temperature == 0 or self.temperature == 1.0:
            next_token = logits.argmax(axis=-1)
        else:
            probs = softmax(logits / self.temperature)
            next_token = sample_top_p(probs, self.top_p)
        logprobs = log_softmax(logits)[next_token]
        hyp.logprob += logprobs
        if next_token == self.eos_id:
            hyp.is_done = True
        hyp.tokens.append(next_token)
        return hyp

    def __call__(
        self,
        audio_feats:np.ndarray,
        max_len:int=50,
        return_multiple:bool=False
    ) -> List[Hypothesis]:
        """
        Performs greedy decoding on audio features.

        Parameters:
            audio_feats (numpy array): Audio features.
            max_len (int): Maximum length of the generated sequence.
            return_multiple (bool): Whether to return multiple hypotheses or the best one.

        Returns:
            Hypothesis: Generated hypothesis.
        """
        self.reset()
        self.inference.encode(audio_feats)
        hyp: Hypothesis = self.inference.get_inits()
        for i in range(1, max_len):
            is_initial = i == 1
            # Retrive the current logits and k_caches and v_caches
            logits, k_cahces, v_caches = self.inference(hyp, initial=is_initial)
            # Update the Hypothesis k_cahces, v_caches 
            hyp.k_caches = k_cahces
            hyp.v_caches = v_caches
            hyp = self.update(logits, hyp)
            if hyp.is_done:
                break
        # Release keys and values caches.    
        hyp.k_caches = None
        hyp.v_caches = None    
        return hyp


class BeamSearchDecoding(Decoding):
    """
    Beam search decoding strategy for sequence generation.

    Attributes:
        inference: The inference instance.
        eos_id: End-of-sequence token ID.
        beam_size: Size of the beam.
        length_penalty: Length penalty factor.
    """

    def __init__(
        self,
        inference,
        eos_id: int,
        beam_size: int = 3,
        length_penalty: float = 1,
        top_p=0.95,
        temperature=1.0,
    ):
        """
        Initializes the BeamSearchDecoding instance.

        Parameters:
            inference: The inference instance.
            eos_id (int): End-of-sequence token ID.
            beam_size (int): Size of the beam.
            length_penalty (float): Length penalty factor.
        """
        super().__init__(inference)
        self.eos_id = eos_id
        self.beam_size = beam_size
        self.length_penalty = MaximumLikelihoodRanker(length_penalty)
        self.temperature = temperature
        self.top_p = top_p

    def update(
        self,
        hyps: List[Hypothesis],
        initial: bool = False,
    ):
        """
        Updates hypotheses based on logits using the beam search strategy.

        Parameters:
            hyps: List of hypotheses.
            initial: Whether it's the initial hyp or not.

        Returns:
            List[Hypothesis]: Updated hypotheses.
        """
        new_beam = []

        for hyp in hyps:
            if hyp.is_done:
                # If the hypothesis is already completed, keep it in the beam
                new_beam.append(hyp)
                continue

            # Get logits for the current hypothesis
            logits, k_caches, v_caches = self.inference(hyp, initial=initial)
            # Apply greedy decode or top p sampling to get the top beam_width candidates
            if self.temperature > 0.0 and self.temperature != 1.0:
                probs = softmax(logits / self.temperature)
                top_indices = sample_top_p(probs, self.top_p, size=self.beam_size)
            else:
                top_indices = np.argsort(logits)[::-1][:self.beam_size]  
            # Apply log softmax normalize then calculate     
            logits = logits - logits.max(axis=-1)    
            sum_logits = np.log(np.sum(np.exp(logits)))
            for idx in top_indices:
                # Create a new hypothesis by extending the current one
                new_tokens = hyp.tokens + [idx]
                #Calculate the log probability
                new_logprob = hyp.logprob + (logits[idx] - sum_logits)
                new_is_done = (idx == self.eos_id)
                # Add the new hypothesis to the beam
                new_beam.append(
                    Hypothesis(
                        tokens=new_tokens,
                        k_caches=k_caches,
                        v_caches=v_caches,
                        logprob=new_logprob,
                        is_done=new_is_done
                    )
                )

        # Sort the beam based on log probabilities
        new_beam = sorted(new_beam, key=lambda h: h.logprob, reverse=True)
        return new_beam[:self.beam_size]

    def __call__(
        self,
        audio_feats: np.ndarray,
        max_len: int = 50,
        return_multiple: bool=False
    ) -> List[Hypothesis]:
        """
        Performs beam search decoding on audio features.

        Parameters:
            audio_feats (numpy array): Audio features.
            max_len (int): Maximum length of the generated sequence.
            return_multiple (bool): Whether to return multiple hypotheses or the best one.

        Returns:
            Hypothesis or List[Hypothesis]: Generated hypothesis or hypotheses.
        """
        self.reset()
        self.inference.encode(audio_feats)
        beam: List[Hypothesis] = [self.inference.get_inits()]
        for i in range(1, max_len):
            is_initial = i == 1
            beam = self.update(
                beam,
                initial=is_initial
            )
            if any(h.is_done for h in beam):
                break
        beam = self.finalize(beam)
        if not return_multiple:
            best_idx = self.length_penalty.rank(beam)
            beam = beam[best_idx]
        return beam
    
    def finalize(self, hyps: List[Hypothesis]):
        """
        Finalizes the decoding process by appending end-of-sequence tokens to hypotheses.

        Parameters:
            hyps: List of hypotheses.

        Returns:
            List[Hypothesis]: Finalized hypotheses.
        """
        for i in range(len(hyps)):
            hyps[i].k_caches = None
            hyps[i].v_caches = None
            if hyps[i].tokens[-1] != self.eos_id:
                hyps[i].tokens.append(self.eos_id)
        return hyps
