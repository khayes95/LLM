import os, warnings, logging
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TQDM_DISABLE'] = '1'
warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'huggingface-space'))

import torch
from pinocchio_local import Pinocchio

LOCAL_ADAPTER = '/scratch/khayes/.cache/huggingface/hub/models--KevinDavidHayes--pinocchio-0.8b/snapshots/de8d63de4b1df94e5d082fc26ba24e653aeb0ed8'
judge = Pinocchio(adapter=LOCAL_ADAPTER, device_map='auto')

examples = [
    ('What is the capital of France?', 'The capital of France is Paris.', 'gpt-5', '', True),
    ('What is the capital of Australia?', 'The capital of Australia is Sydney.', 'gpt-5-mini', '', False),
    ('What is the capital of Australia?', 'The capital of Australia is Canberra.', 'gpt-5', '', True),
    ('What is 2+2?', '4', 'gpt-5', '', True),
    ('What is 2+2?', 'The answer is 4.', 'gpt-5', '', True),
    ('What is 2+2?', '5', 'gpt-5', '', False),
    ('What is 2+2?', 'The answer is 5.', 'gpt-5', '', False),
    ('Who wrote To Kill a Mockingbird?', 'Harper Lee.', 'gpt-5', 'triviaqa', True),
    ('Who wrote To Kill a Mockingbird?', 'Mark Twain.', 'gpt-5-mini', 'triviaqa', False),
    ('What is the integral of x^2?', '(x^3)/3 + C', 'claude-4-sonnet', 'math', True),
    ('What is the integral of x^2?', 'x^3 + C', 'gpt-5-mini', 'math', False),
    ('What color is the sky?', 'The sky is blue.', 'gpt-5', '', True),
    ('What color is the sky?', 'The sky is green.', 'gpt-5', '', False),
    ('How many legs does a dog have?', 'A dog has four legs.', 'gpt-5', '', True),
    ('How many legs does a dog have?', 'A dog has six legs.', 'gpt-5', '', False),
]

print(f"{'GT':<6} {'P(cor)':>6}  {'Question':<42} {'Answer':<35} Check")
print('-' * 105)
for q, a, model, bench, correct in examples:
    score = judge.score(question=q, answer=a, source_model=model, benchmark=bench)
    ok = 'OK' if (score > 0.5) == correct else 'WRONG'
    gt = 'TRUE' if correct else 'FALSE'
    print(f"{gt:<6} {score:>6.3f}  {q:<42} {a:<35} [{ok}]")
