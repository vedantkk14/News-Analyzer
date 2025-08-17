# from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# class NewsSummarizer:
#     def __init__(self, local_path="./models/pegasus-xsum"):
#         tokenizer = AutoTokenizer.from_pretrained(local_path)
#         model = AutoModelForSeq2SeqLM.from_pretrained(local_path)
#         self.summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

#     def summarize(self, text, min_len=5, max_len=50):
#         summary = self.summarizer(
#             text,
#             min_length=min_len,
#             max_length=max_len,
#             do_sample=False
#         )
#         return summary[0]['summary_text']




from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

class NewsSummarizer:
    def __init__(self, local_path="./models2/bart-large-cnn"):
        tokenizer = AutoTokenizer.from_pretrained(local_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(local_path)
        self.summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

    def summarize(self, text, min_len=10, max_len=100):
        summary = self.summarizer(
            text,
            min_length=min_len,
            max_length=max_len,
            do_sample=False
        )
        return summary[0]['summary_text']




