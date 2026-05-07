# create class for text preprocessing
import pandas as pd
import re

# load dataset
df = pd.read_csv('datasets/raw_tickets.csv')

# print first 5 rows of the dataset
if df.empty:
    print("Dataset is empty.")
else:
    print("Dataset loaded successfully.")
    print(df.head())

dataset = df['DESKRIPSI'].tolist()

# 0. remove line klien: or client: or clien: with or without space before ':' 
preprocessed_data = [re.sub(r'(?im)^(klien|client)\s*:\s*.*$\n?', '', desc, flags=re.IGNORECASE) for desc in dataset]

# 1. replace \n with '. '
preprocessed_data = [re.sub(r'\n+', '. ', desc) for desc in preprocessed_data]

# 2. case folding
preprocessed_data = [desc.lower() for desc in preprocessed_data]

# 3. masking url with tautan
preprocessed_data = [re.sub(r'http\S+', 'tautan', desc) for desc in preprocessed_data]

# 4. masking account mention with nama orang
preprocessed_data = [re.sub(r'@\w+', 'nama orang', desc) for desc in preprocessed_data]

# 5. masking email with surel
preprocessed_data = [re.sub(r'\S+@\S+', 'surel', desc) for desc in preprocessed_data]

# 6. remove duplicate characters such as "hiiii" to "hii"
preprocessed_data = [re.sub(r'(.)\1{2,}', r'\1\1', desc) for desc in preprocessed_data]

# 7. remove emojis 
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+", flags=re.UNICODE)
preprocessed_data = [emoji_pattern.sub(r'', desc) for desc in preprocessed_data]

# print the preprocessed data, new line every description, only print the first 5 descriptions
for desc in preprocessed_data[:5]:
    print(desc + '\n')

# save the preprocessed data to a new txt file
with open('datasets/preprocessed_tickets.txt', 'w', encoding='utf-8') as f:
    for desc in preprocessed_data:
        f.write(desc + '\n')

# print message that the preprocessed data has been saved
print("Preprocessed data has been saved to 'datasets/preprocessed_tickets.txt'.")