# Siamese-network-based-handwriting-verification-from-questioned-document-perspective

Requirements
Python 3.7 cycler 0.11.0 future 1.0.0 kiwisolver 1.4.5 matplotlib 3.3.3 numpy 1.18.1 pandas 1.1.4 Pillow 9.5.0 pip 10.0.1 pyparsing 3.1.2 python-dateutil 2.9.0.post0 pytz 2024.1 setuptools 39.0.1 six 1.16.0 torch 1.5.0 torchvision 0.6.0 typing-extensions 4.7.1

First install python version 3.7 then pip install all the other packages as per the versioning or else it wont work. The code requiere a GPU and you need to install CUDA for NVIDIA. I have used version 11.8 for the pytorch (torch) to work.

Train the model on one set of character like only on one character or letter for both orignal and non original samples. Take only one set of samples from one write and one set from another writer and then train the model. Then test it with one set being the same a training samples another from different writer or different writing of the same writer. Then you will get the euclidean distance then paste it in output.csv as column wise for as shown in the output.csv already. Then run the "run this for plotting.py" file to visualoizes the distance in a plot. If the distance is in range of the original samples then the writen character is probably similar or else if the questioned pair is having more distance than the original pair then it might be fake.

To run:
First put the samples in tr folder 1 for original training set in folder 1f for fake training set. Then add the path in config.py file and run the program. once the program ends it will save the trained model as model1.pth incontent folder. Then save the images you placed in tr folder 1 inside te folder 1 folder. The questioned samples place in the te folder 1f folder. Then run test.py after correcting the path inside that file. The program will out the euclidean distance. Then save it in output.csv as the format in that file in each column. Then run "run this for plotting.py" to visualize the distance in scatter plot.
