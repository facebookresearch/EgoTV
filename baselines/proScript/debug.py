from transformers import T5Tokenizer, T5ForConditionalGeneration
t5_model = T5ForConditionalGeneration.from_pretrained("t5-large")
tokenizer = T5Tokenizer.from_pretrained("t5-large")

i1 = 'apple is picked, then heated, then cleaned in a SinkBasin, then sliced'
g1 = 'digraph graph {"Step 1 heat apple";"Step 2 clean apple";"Step 3 slice apple";"Step 1 heat apple" -> "Step 2 clean apple";"Step 2 clean apple" -> "Step 3 slice apple";}'
i2 = 'apple is sliced after heating'
g2 = 'digraph graph {"Step 1 heat apple";"Step 2 slice apple";"Step 1 heat apple" -> "Step 2 slice apple";}'
i3 = 'apple is picked, heated, then sliced'
g3 = 'digraph graph {"Step 1 heat apple";"Step 2 slice apple";"Step 1 heat apple" -> "Step 2 slice apple";}'
prompt_input = '{} => {} \n {} => {} \n {} => '.format(i1, g1, i2, g2, i3)
prompt_ids = tokenizer(prompt_input, return_tensors="pt").input_ids
outputs = t5_model.generate(prompt_ids)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
