# TODO List
Experiment design

2017.8.31
1. - is not cared.
2. there are some countries' transliteration is not really standard
3. no more findings


2017.9.2 - sad face. rest yesterday
1. map words with pronunciations aka phonetic symbols
2. Using cmudict-0.7b phones. dataset is in http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/
3. Using http://www.speech.cs.cmu.edu/tools/lextool.html to generate corresponding pronunciations
4. As the tool only allow limited size words as input. So use script to seperate the whole file into 6 parts and input
5. Combine all results into one using file
6. change format to '(english_word, chinese word)': location: if_has, pronunciation:__, etc:___ 

@fix last edit for easy use: change saving data format to '(english_word, chinese word)': location: if_has, pronunciation:__, etc:___ 

7. the json file is too large when processing. for easy use. change it to normal txt format. always save last column to be location(if has). previous format is english_word, chinese_word, pronounciation, location(if has)