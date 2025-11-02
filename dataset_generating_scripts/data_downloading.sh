mkdir RawData && cd RawData


mkdir AirQuality && cd AirQuality
wget http://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip
unzip PRSA2017_Data_20130301-20170228.zip
cd ..

mkdir Electricity && cd Electricity
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip
unzip LD2011_2014.txt.zip
cd ..

mkdir ETT && cd ETT
wget https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv
cd ..
