import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def optimize_model_with_dpo(preference_data, model_name="gpt2"):
    """
    Optimize the model using DPO based on the feedback data.
    The feedback data contains the prompt, correct answer, and rejected answer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for index, row in preference_data.iterrows():
        prompt = row['prompt']
        correct_answer = row['correct_answer']
        rejected_answer = row['rejected_answer']

        # Only optimize on correct answers
        if correct_answer is not None:
            inputs = tokenizer(prompt + correct_answer, return_tensors="pt")
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Save the updated model
    model.save_pretrained("./models/updated_dpo_model")
    print("Model optimization complete and saved as updated_dpo_model.")
