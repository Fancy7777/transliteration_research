# Transliteration Research Project

# Research Result
Approaches: 
1. Rule Based: Advantage: Fast, Straight forward Disadvantage: Hard to distinguish right matched word but only based on rules
2. Statistical based method: Improve rule based method but using statistics to help. Depend on dataset mainly.

Main methods(Most people used): combine rule based and statistical method
Procedure:
1. English <==> English Phonetic symbols
2. English Phonetic symbols <==> Chinese Pinyin
3. Chinese Pinyin <==> Transliteration

Things may need improve:
1. Chinese Pinyin dataset always do not have tone data. And one simple Pinyin could match several words
E.g wo = 我，窝，握 etc
2. English Phonetic symbols match with Chinese Pinyin
3. Could engage with other techniques such as web mining

# TODO List
Experiment design

1. Decide using dataset include: CMU Pronunciation dicitonary, LDC2005T34, LDC2013T06
2. Preprocessing data to suitable format
3. Decide evaluation method
4. Start normal rule based method to construct baseline
5. Use Statistical method to Try language model to see if can be improved.
6. Try Other methods include RNN, RNN with LSTM, Seq2seq, word2vec etc

Do experiement with analysis and adjust plans when necessary. Add other improve method when the accuracy is sastisfactory.



2017/8/31:
1. ldc_propernames_other_ec_v1.beta_transformed.txt need to filter out some not tranliteration's data
2. Same Chinese transliteration choose USA version
3. Some are just dictionary based. Need to use a dictionary to filter

4. First filter out the same tranliteration. 
5. ldc_propernames_place_ec_v1.beta_transformed.txt filter out only English speaking country. Other country is not neccesary. English spearking country include: USA, UK, Canada, South Africa, Australia, New Zealand, Pakistan, India, Ireland,Nigeria，Philippines,Malaysia,Uganda,Ghana, Cameroon,Zimbabwe,Zambia,Rwanda,sierra leone,Papua New Guinea,Puerto Rico,Jamaica,Namibia


Good dataset: ldc_propernames_people_ec_v1.beta_transformed.txt,  