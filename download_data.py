import gdown
import zipfile
import os

# Google Drive file ID for data.zip
file_id = '1Xb9q79ZbSVSKErvazg7BQEKXvuiYTzJ-'
output = 'data.zip'

# Download the file from Google Drive
print('Downloading data.zip from Google Drive...')
gdown.download(f'https://drive.google.com/uc?id={file_id}', output, quiet=False)

# Unzip the file
print('Extracting data.zip...')
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall('data')

# Remove the zip file after extraction
os.remove(output)

print("Data downloaded and extracted to the 'data/' folder.") 