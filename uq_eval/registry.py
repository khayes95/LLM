from __future__ import annotations

from typing import Any

from .benchmarks.base import BaseBenchmark
from .models.base import BaseModelClient

from .benchmarks.dummy_qa import DummyQABenchmark
from .benchmarks.sanity_mcq import SanityMCQBenchmark
from .benchmarks.sanity_math import SanityMathBenchmark
from .benchmarks.sanity_unanswerable import SanityUnanswerableBenchmark
from .benchmarks.sanity_long_context import SanityLongContextBenchmark
from .benchmarks.jsonl_qa import JsonlQABenchmark
from .benchmarks.gpqa import GPQABenchmark
from .benchmarks.simpleqa import SimpleQABenchmark
from .benchmarks.bbeh import BBEHBenchmark
from .benchmarks.hle import HLEBenchmark
from .benchmarks.healthbench import HealthBenchBenchmark
from .benchmarks.tutorbench import TutorBenchBenchmark
from .benchmarks.multinrc import MultiNRCBenchmark
from .benchmarks.ether0 import Ether0Benchmark
from .benchmarks.omnimath import OmniMathBenchmark
from .benchmarks.math_benchmark import MATHBenchmark
from .benchmarks.mmlu_pro import MMLUProBenchmark
from .benchmarks.mgsm import MGSMBenchmark
from .benchmarks.multichallenge import MultiChallengeBenchmark
from .benchmarks.bigcodebench import BigCodeBenchBenchmark
from .benchmarks.livecodebench import LiveCodeBenchBenchmark
from .benchmarks.swebench import SWEBenchBenchmark
from .benchmarks.oolong import OolongBenchmark
from .benchmarks.arc import ARCBenchmark
from .benchmarks.drop import DROPBenchmark
from .benchmarks.gsm8k import GSM8KBenchmark
from .benchmarks.triviaqa import TriviaQABenchmark
from .benchmarks.hellaswag import HellaSwagBenchmark
from .benchmarks.winogrande import WinograndeBenchmark
from .benchmarks.livebench import LiveBenchBenchmark
from .benchmarks.longbench import LongBenchBenchmark, LongBenchV2Benchmark
from .benchmarks.aime import AIMEBenchmark
from .benchmarks.mmlu import MMLUBenchmark
from .benchmarks.humaneval import HumanEvalBenchmark, HumanEvalPlusBenchmark
from .benchmarks.mbpp import MBPPBenchmark
from .benchmarks.naturalqa import NaturalQuestionsBenchmark
from .benchmarks.boolq import BoolQBenchmark
from .benchmarks.arc_agi import ARCAGIBenchmark
from .benchmarks.babilong import BABILongBenchmark
from .benchmarks.chembench import ChemBenchBenchmark
from .benchmarks.prbench import PRBenchBenchmark
from .benchmarks.mmmu import MMMUBenchmark
from .benchmarks.charxiv import CharXivBenchmark
from .benchmarks.erqa import ERQABenchmark
from .benchmarks.mathvista import MathVistaBenchmark
from .benchmarks.mathverse import MathVerseBenchmark
from .benchmarks.hallusionbench import HallusionBenchBenchmark
from .benchmarks.mathvision import MathVisionBenchmark
from .benchmarks.mmstar import MMStarBenchmark
from .benchmarks.realworldqa import RealWorldQABenchmark
from .benchmarks.aokvqa import AOKVQABenchmark
from .benchmarks.vizwiz import VizWizBenchmark
from .benchmarks.mmvet import MMVetBenchmark
from .benchmarks.vsr import VSRBenchmark

from .models.openai_client import OpenAIResponsesClient
from .models.chat_completions_http_client import ChatCompletionsHTTPClient
from .models.internvl_client import InternVLClient
from .models.qwen3_vl_client import Qwen3VLClient
from .models.qwen3_vl_vllm_client import Qwen3VLvLLMClient


# Factory function for HLE multimodal variant
def _hle_multimodal(**kw):
    return HLEBenchmark(include_images=True, text_only=False, **kw)


_BENCH_REGISTRY: dict[str, type[BaseBenchmark]] = {
    "dummy_qa": DummyQABenchmark,
    "sanity_mcq": SanityMCQBenchmark,
    "sanity_math": SanityMathBenchmark,
    "sanity_unanswerable": SanityUnanswerableBenchmark,
    "sanity_long_context": SanityLongContextBenchmark,
    "jsonl_qa": JsonlQABenchmark,
    "gpqa": GPQABenchmark,
    "simpleqa": SimpleQABenchmark,
    "bbeh": BBEHBenchmark,
    "hle": HLEBenchmark,
    "hle_multimodal": _hle_multimodal,
    "healthbench": HealthBenchBenchmark,
    "tutorbench": TutorBenchBenchmark,
    "multinrc": MultiNRCBenchmark,
    "ether0": Ether0Benchmark,
    "omnimath": OmniMathBenchmark,
    "math": MATHBenchmark,
    "mmlu_pro": MMLUProBenchmark,
    "mgsm": MGSMBenchmark,
    "multichallenge": MultiChallengeBenchmark,
    "bigcodebench": BigCodeBenchBenchmark,
    "livecodebench": LiveCodeBenchBenchmark,
    "swebench": SWEBenchBenchmark,
    "oolong": OolongBenchmark,
    "arc": ARCBenchmark,
    "drop": DROPBenchmark,
    "gsm8k": GSM8KBenchmark,
    "triviaqa": TriviaQABenchmark,
    "hellaswag": HellaSwagBenchmark,
    "winogrande": WinograndeBenchmark,
    "livebench": LiveBenchBenchmark,
    "longbench": LongBenchBenchmark,
    "longbench_v2": LongBenchV2Benchmark,
    "aime": AIMEBenchmark,
    "mmlu": MMLUBenchmark,
    "humaneval": HumanEvalBenchmark,
    "humanevalplus": HumanEvalPlusBenchmark,
    "mbpp": MBPPBenchmark,
    "naturalqa": NaturalQuestionsBenchmark,
    "boolq": BoolQBenchmark,
    "arc_agi": ARCAGIBenchmark,
    "babilong": BABILongBenchmark,
    "chembench": ChemBenchBenchmark,
    "prbench": PRBenchBenchmark,
    "mmmu": MMMUBenchmark,
    "charxiv": CharXivBenchmark,
    "erqa": ERQABenchmark,
    "mathvista": MathVistaBenchmark,
    "mathverse": MathVerseBenchmark,
    "hallusionbench": HallusionBenchBenchmark,
    "mathvision": MathVisionBenchmark,
    "mmstar": MMStarBenchmark,
    "realworldqa": RealWorldQABenchmark,
    "aokvqa": AOKVQABenchmark,
    "vizwiz": VizWizBenchmark,
    "mmvet": MMVetBenchmark,
    "vsr": VSRBenchmark,
}

_MODEL_REGISTRY: dict[str, type[BaseModelClient]] = {
    "openai": OpenAIResponsesClient,
    "chat_http": ChatCompletionsHTTPClient,
    "internvl": InternVLClient,
    "qwen3_vl": Qwen3VLClient,
    "qwen3_vl_vllm": Qwen3VLvLLMClient,
}


def load_benchmark(name: str, **kwargs: Any) -> BaseBenchmark:
    name = name.strip()
    if name not in _BENCH_REGISTRY:
        raise KeyError(f"Unknown benchmark: {name}. Available: {sorted(_BENCH_REGISTRY)}")
    return _BENCH_REGISTRY[name](**kwargs)


def load_model_client(backend: str, **kwargs: Any) -> BaseModelClient:
    backend = backend.strip()
    if backend not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model backend: {backend}. Available: {sorted(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[backend](**kwargs)
