# # %% cell1
# import pandas as pd
# data = pd.read_csv("guj_Gujr.tsv", sep="\t")

# print(data.head(10))

# for column in data.columns:
#     print(f"{column}: {data[column].iloc[0]}")
#     # store only the first row in a new CSV file
#     # Create a new DataFrame without the first and second columns
# data_to_write = data.iloc[2:].drop(data.columns[:2], axis=1)
# data_to_write.to_csv("all_rows_except_first_second.txt", index=False)
# data.iloc[0:1].to_csv("tep.csv", index=False)

from tokenizer.tokenizer import Tokenizer

special_token = ["<en|gu>", "<gu|en>","<en>", "<gu>"]
tokenizer = Tokenizer(special_token)

tokenizer.load("./data/tokenizer.model")

print(tokenizer.encode("<en|gu> વપરાશકર્તા દ્વારા આપવામાં આવતી ઇનપુટ સામાન્ય જમણે-થી-ડાબે ફોર્મેટને Hellબદલે ડાબેથી જમણે દાખલ થાય છે",out_type="str"))
print(tokenizer.decode(tokenizer.encode("વપરાશકર્તા દ્વારા આપવામાં આવતી ઇનપુટ સામાન્ય જમણે-થી-ડાબે ફોર્મેટને બદલે ડાબેથી જમણે દાખલ થાય છે")))
print(tokenizer.encode("મારુ નામ ચૈત છે"))
