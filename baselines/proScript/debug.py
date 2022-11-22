from transformers import T5Tokenizer, T5ForConditionalGeneration
t5_model = T5ForConditionalGeneration.from_pretrained("t5-11b")
tokenizer = T5Tokenizer.from_pretrained("t5-11b")

# i1 = 'apple is picked, then heated, then cleaned in a SinkBasin, then sliced'
# g1 = 'digraph graph {"Step 1 heat apple";"Step 2 clean apple";"Step 3 slice apple";"Step 1 heat apple" -> "Step 2 clean apple";"Step 2 clean apple" -> "Step 3 slice apple";}'
# i2 = 'apple is sliced after heating'
# g2 = 'digraph graph {"Step 1 heat apple";"Step 2 slice apple";"Step 1 heat apple" -> "Step 2 slice apple";}'
# i3 = 'apple is picked, heated, then sliced'
# g3 = 'digraph graph {"Step 1 heat apple";"Step 2 slice apple";"Step 1 heat apple" -> "Step 2 slice apple";}'

# i1 = 'a sliced tomato'
# g1 = 'Step 1 slice tomato'
# i2 = 'apple is cooled in a Fridge'
# g2 = 'Step 1 cool apple'
# i3 = 'potato is sliced after heating'
# g3 = 'Step 1 heat potato; Step 2 slice potato'
# i4 = 'tomato is picked, heated, then sliced'
# g4 = 'Step 1 heat tomato, Step 2 slice tomato'
# i5 = 'lettuce is sliced and cleaned'
# g5 = 'Step 1 slice lettuce, Step 2 clean lettuce'
# i6 = 'apple is picked, then heated, then cleaned in a SinkBasin, then sliced'
# g6 = 'Step 1 heat apple, Step 2 clean apple, Step 3 slice apple'
# i7 = 'potato is cooled in a Fridge then cleaned in a SinkBasin'
# g7 = 'Step 1 cool apple, Step 2 clean apple'
# i8 = 'apple is heated before slicing and cleaning'
# g8 = 'Step 1 heat apple, Step 2 slice apple, Step 3 clean apple'

i1 = 'a sliced tomato'
g1 = 'slice'

i2 = 'apple is cooled in a Fridge'
g2 = 'cool'

i3 = 'potato is heated, then sliced'
g3 = 'heat, slice'
# i4 = 'tomato is picked, heated, then sliced'
# g4 = 'heat tomato, slice tomato'
i5 = 'lettuce is sliced and cleaned'
g5 = 'slice, clean'
# i6 = 'apple is picked, then heated, then cleaned in a SinkBasin, then sliced'
# g6 = 'heat apple, clean apple, slice apple'
i7 = 'potato is cooled, then cleaned'
g7 = 'cool, clean'
# i8 = 'apple is heated before slicing and cleaning'
# g8 = 'heat apple, slice apple, clean apple'
#
i9 = 'tomato is cooled, then sliced'
i10 = 'tomato is cooled and placed after slicing'

# prompt_input = '{} => {} \n {} => {} \n {} => {} \n {} => {} \n {} => {} \n {} => {} \n {} => {} \n {} => {} ' \
#                '\n {} => '.format(i1, g1, i2, g2, i3, g3, i4, g4, i5, g5, i6, g6, i7, g7, i8, g8, i9)
prompt_input = '{} = {}; {} = {}; {} = {}; {} = {}; {} ='.format(i1, g1, i2, g2, i3, g3, i7, g7, i9)
print(prompt_input)
prompt = tokenizer(prompt_input, return_tensors="pt")
outputs = t5_model.generate(input_ids=prompt["input_ids"], attention_mask=prompt["attention_mask"])
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
