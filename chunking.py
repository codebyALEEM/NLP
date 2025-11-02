import nltk
from nltk import pos_tag,word_tokenize,RegexpParser

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

text = "full stack datascience,generative ai,llm model keep increase by different company"

tokens = word_tokenize(text)

tagged_tokens = pos_tag(tokens)

chunk_grammer = r"""
 NP: {<DT>?<JJ>*<NN>}
 VP: {<VB.*><NP|PP>*}
 PP: {<IN><NP>}
 """
 
chunk_parser = RegexpParser(chunk_grammer)
 
chunked = chunk_parser.parse(tagged_tokens)

print(chunked) 

chunked.draw()