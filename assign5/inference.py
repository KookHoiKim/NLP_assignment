from typing import List
import json

import torch
import torch.nn.functional as F

### YOUR LIBRARIES HERE

### END YOUR LIBRARIES

from model import BertForSquad
from dataset import squad_features, SquadDataset, SquadFeatureDataset
from evaluation import evaluate

from transformers import BertTokenizerFast
from tqdm import trange

##### CHANGE BELOW OPTIONS IF IT IS DIFFENT FROM YOUR OPTIONS #####
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
bert_type = 'bert-base-uncased' 

def inference_start_end(
    start_probs: torch.Tensor,
    end_probs: torch.Tensor,
    context_start_pos: int,
    context_end_pos: int
):
    """ Inference fucntion for the start and end token position.

    Find the start and end positions of the answer which maximize
    p(start, end | context_start_pos <= start <= end <= context_end_pos)

    Note: assume that p(start) and p(end) are independent.

    Hint: torch.tril or torch.triu function would be helpful.

    Arguments:
    start_probs -- Probability tensor for the start position
                    in shape (sequence_length, )
    end_probs -- Probatility tensor for the end position
                    in shape (sequence_length, )
    context_start_pos -- Start index of the context
    context_end_pos -- End index of the context
    """
    assert start_probs.sum().allclose(torch.scalar_tensor(1.))
    assert end_probs.sum().allclose(torch.scalar_tensor(1.))

    ### YOUR CODE HERE (~6 lines)
    start_pos: int = None
    end_pos: int = None

    ### END YOUR CODE

    return start_pos, end_pos

def inference_answer(
    question: str,
    context: str,

    input_ids: List[int],
    token_type_ids: List[int],
    start_pos: int,
    end_pos: int,

    tokenizer: BertTokenizerFast
) -> str:
    """ Inference fucntion for the answer.

    Because the tokenizer lowers the capital letters and splits punctuation marks,
    you may get wrong answer words if you detokenize it directly.
    For example, if you encode "$5.000 Dollars" and decode it, you get different words from the orignal.

    "$5.00 USD" --(Tokenize)--> ["$", "5", ".", "00", "usd"] --(Detokenize)--> "$ 5. 00 usd"

    Thus, you should find the original words in the context by the start and end token positions of the answer.
    Implement the function inferencing the answer from the context and the answer token postion.

    Note 1: We have already implmented direct decoding so you can skip this problem if you want.

    Note 2: When we implement squad_feature, we have arbitrarily split tokens if the answer is a subword,
            so it is very tricky to extract the original word by start_pos and end_pos.`
            However, as None is entered into the answer when evaluating,
            you can assume the word tokens follow general tokenizing rule in this problem.
            In fact, the most appropriate solution is storing the character index when tokenizing them.

    Hint: You can find a simple solution if you carefully search the documentation of the transformers library.
    Library Link: https://huggingface.co/transformers/index.html

    Arguments:
    question -- Question string
    context -- Context string

    input_ids -- Input ids
    token_type_ids -- Token type ids
    start_pos -- Predicted start token position of the answer
    end_pos -- Predicted end token position of the answer

    tokenizer -- Tokenizer to encode and decode the string

    Return:
    answer -- Answer string
    """
    ### YOUR CODE HERE (~4 lines)
    answer = input_ids[start_pos: end_pos+1]
    answer: str = tokenizer.decode(answer)

    ### END YOUR CODE

    return answer

def inference_model(
    model: BertForSquad,
    tokenizer: BertTokenizerFast,

    context: str,
    question: str,
    input_ids: List[int],
    token_type_ids: List[int]
) -> str:
    """ Inferene function with the model 
    Because we don't know how your model works, we can't not infer the answer from your model.
    Implement inference process for you model.
    Please use inference_start_end and inference_answer functions you have implemented
    
    Argumentes:
    model -- Model you have trained.
    tokenizer -- Tokenizer to encode and decode the string
    context -- Context string
    question -- Question string
    input_ids -- Input ids
    token_type_dis -- Token type ids

    Return:
    answer -- Answer string
    """
    ### YOUR CODE HERE
    anwser: str = None

    ### END YOUR CODE

    return answer 

#############################################
# Testing functions below.                  #
#############################################

def test_inference_start_end(tokenizer):
    print("======Start End Inference Test Case======")
    
    # First test
    input_tokens = ['[CLS]', 'this', 'is', 'a', 'question', '.', '[SEP]', 'this', 'is', 'an', 'answer', '.', '[SEP]']
    start_probs  = [    0.0,    0.0,  0.0, 0.0,        0.0, 0.0,     0.0,    0.0,  0.1,  0.8,      0.1, 0.0,     0.0]
    end_probs    = [    0.0,    0.0,  0.0, 0.0,        0.0, 0.0,     0.0,    0.0,  0.0,  0.1,      0.8, 0.1,     0.0]
    context_start_pos = input_tokens.index('[SEP]') + 1
    context_end_pos = len(input_tokens) - 2

    start_probs = torch.Tensor(start_probs)
    end_probs = torch.Tensor(end_probs)
    start_pos, end_pos = inference_start_end(start_probs, end_probs, context_start_pos, context_end_pos)
    answer = input_tokens[start_pos: end_pos+1] 

    assert answer == ['an', 'answer'], \
        "Your infered position is different from the expected position."
    
    print("The first test passed!")

    # Second test
    input_tokens = ['[CLS]', 'this', 'is', 'a', 'question', '.', '[SEP]', 'this', 'is', 'an', 'answer', '.', '[SEP]']
    start_probs  = [    0.0,    0.0,  0.0, 0.0,        0.0, 0.0,     0.0,    0.0,  0.1,  0.8,      0.1, 0.0,     0.0]
    end_probs    = [    0.0,    0.0,  0.0, 0.0,        0.0, 0.0,     0.0,    0.0,  0.6,  0.1,      0.3, 0.0,     0.0]
    context_start_pos = input_tokens.index('[SEP]') + 1
    context_end_pos = len(input_tokens) - 2

    start_probs = torch.Tensor(start_probs)
    end_probs = torch.Tensor(end_probs)
    start_pos, end_pos = inference_start_end(start_probs, end_probs, context_start_pos, context_end_pos)
    answer = input_tokens[start_pos: end_pos+1] 

    assert answer == ['an', 'answer'], \
        "Your infered position is different from the expected position."
    print("The second test passed!")

    # third test
    input_tokens = ['[CLS]', 'this', 'is', 'a', 'question', '.', '[SEP]', 'this', 'is', 'an', 'answer', '.', '[SEP]']
    start_probs  = [    0.0,    0.0,  0.0, 0.0,        0.0, 0.0,     0.0,    0.1,  0.2,  0.3,      0.1, 0.3,     0.0]
    end_probs    = [    0.0,    0.0,  0.0, 0.0,        0.0, 0.0,     0.0,    0.4,  0.2,  0.1,      0.2, 0.1,     0.0]
    context_start_pos = input_tokens.index('[SEP]') + 1
    context_end_pos = len(input_tokens) - 2

    start_probs = torch.Tensor(start_probs)
    end_probs = torch.Tensor(end_probs)
    start_pos, end_pos = inference_start_end(start_probs, end_probs, context_start_pos, context_end_pos)
    answer = input_tokens[start_pos: end_pos+1] 

    assert answer == ['an', 'answer'], \
        "Your infered position is different from the expected position."
    print("The third test passed!")

    # forth test
    input_tokens = ['[CLS]', 'this', 'is', 'a', 'question', '.', '[SEP]', 'this', 'is', 'an', 'answer', '.', '[SEP]']
    start_probs  = [    0.0,    0.0,  0.0, 0.0,        0.0, 0.3,     0.3,    0.0,  0.1,  0.2,      0.1, 0.0,     0.0]
    end_probs    = [    0.0,    0.0,  0.0, 0.0,        0.0, 0.0,     0.0,    0.0,  0.2,  0.0,      0.2, 0.0,     0.6]
    context_start_pos = input_tokens.index('[SEP]') + 1
    context_end_pos = len(input_tokens) - 2

    start_probs = torch.Tensor(start_probs)
    end_probs = torch.Tensor(end_probs)
    start_pos, end_pos = inference_start_end(start_probs, end_probs, context_start_pos, context_end_pos)
    answer = input_tokens[start_pos: end_pos+1] 

    assert answer == ['an', 'answer'], \
        "Your infered position is different from the expected position."
    print("The forth test passed!")

    print("All 4 test passed!")

def test_inference_answer(tokenizer):
    print("======Answer Inference Test Case======")

    # First test
    context = "The example answer was $5.00 USD."
    question = "What was the answer?"
    answer = "$5.00 USD"
    start_pos = context.find(answer)

    input_ids, token_type_ids, start_pos, end_pos = squad_features(context, question, answer, start_pos, tokenizer)
    prediction = inference_answer(question, context, input_ids, token_type_ids, start_pos, end_pos, tokenizer)
    
    if prediction == "$ 5. 00 usd":
        print("Skip the test. You get no score.")
        return

    assert prediction == answer, \
        "Your answer is different from the expected answer."

    print("The first test passed!")

    # Second test
    context = "The speed of the light is 299,794,458 m/s."
    question = "What is the speed of the light?"
    answer = "299,794,458 m/s"
    start_pos = context.find(answer)

    input_ids, token_type_ids, start_pos, end_pos = squad_features(context, question, answer, start_pos, tokenizer)
    prediction = inference_answer(question, context, input_ids, token_type_ids, start_pos, end_pos, tokenizer)

    assert prediction == answer, \
        "Your answer is different from the expected answer."

    print("The second test passed!")

    print("All 2 test passed!")

#############################################
# Analysis functions below.                  #
#############################################

def qualitative_analysis(tokenizer, model):
    print("======Qualitative Analysis======")
    # Select your sample
    question = "How much time did it take to make this assignment?"
    context = '''Making an assignment is a very time-consuming task. 
         Assignments are created in the order of preliminary investigation, code design, review and documentation. 
         Each step takes more than 3 days, and coding especially takes much than other steps.
         Therefore, it takes roughly two weeks to make a single assignment, and this assignment also took that much.
         Thus, please give praise and encouragement for TAs who struggle hard in the invisible place. ;) '''
    plausible_answers = ["two weeks", "roughly two weeks"]
    start_pos = context.find(plausible_answers[0])
    input_ids, token_type_ids, _, _ = squad_features(context, question, plausible_answers[0], start_pos, tokenizer)

    prediction = inference_model(model, tokenizer, context, question, input_ids, token_type_ids)

    print(f"Context: {context}")
    print(f"Question: {question}")
    print(f"Plausible Answers: {plausible_answers}")
    print(f"Prediction: {prediction}")

def quantative_analysis(tokenizer, model):
    print("======Quantitative Analysis======")
    dataset = SquadDataset('data/dev-v1.1-TA.json')
    dataset = SquadFeatureDataset(dataset, bert_type=bert_type, lazy=True, return_sample=True, eval=True)

    answers = dict()

    for index in trange(len(dataset), desc="Answering"):
        (input_ids, token_type_ids, _, _), sample = dataset[index]
        answers[sample['id']] = \
                inference_model(model, tokenizer, sample['context'], sample['question'], input_ids, token_type_ids)

    with open('dev-v1.1-TA-answers.json', mode='w') as f:
        json.dump(answers, f)
    
    with open('data/dev-v1.1-TA.json', mode='r') as f:
        dataset = json.load(f)['data']

    results = evaluate(dataset, answers)
    print(f"Exact Match: {results['exact_match']}. This should be upper than 60.0. TA score: 75.2")
    print(f"F1 score: {results['f1']}. This should be upper than 70.0. TA score: 83.9")

if __name__ == "__main__":
    tokenizer = BertTokenizerFast.from_pretrained(bert_type)

    test_inference_start_end(tokenizer)
    test_inference_answer(tokenizer)

    model = BertForSquad.from_pretrained('./checkpoint')
    model.to(device)
    model.eval()

    qualitative_analysis(tokenizer, model)
    quantative_analysis(tokenizer, model)
