# Bert-Inferencing
Inferencing time calculation using BERT for Sentiment Analysis of Tweets
## Steps to Run
#Set Up environment
sudo apt-get install cpufrequtils 
echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils 
sudo systemctl disable ondemand.service 
sudo apt-get install google-perftools 
# Run File
Run run_updated.sh directly (in case of tensorflow cpu)
Uncomment line #8 and comment line #7 in case of tensorflow-gpu installation
