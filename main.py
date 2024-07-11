from transformers import AutoTokenizer, AutoModel

def downloadmodel(model):

    #model_user_text,model_name_text = model.split("/")
    

    tokenizer = AutoTokenizer.from_pretrained(model,token='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')
    model = AutoModel.from_pretrained(model,token='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')

    tokenizer.save_pretrained("Downloads/")
    model.save_pretrained("Downloads/")



if __name__ == "__main__":
    downloadmodel('sentence-transformers/all-mpnet-base-v2')