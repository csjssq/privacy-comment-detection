1、Using the sample method in the Twarc2 toolkit, the v2 node sample was called
   twarc2 sample --limit 5000 sampleoutput.jsonl

2、Using twarc2 plugins to extract csv files from the sample output
   twarc2 csv --output-columns conversation_id,text sampleoutput.jsonl sampleoutput.csv

3、filter lang=en and select id=conversation_id 
   ref to getthefirstcolumn.py
   twarc2 conversations --conversation-limit 1 ids.txt sampleoutput_id=conversation.jsonl
   twarc2 csv --output-columns conversation_id,text sampleoutput_id=conversation.jsonl sampleoutput_id=conversation.csv

4、select id!=conversation_id and hydrate the original tweet using conversation_id
   ref to getthefirstcolumn.py == to match; != to not match
   twarc2 hydrate ids.txt sampleoutput_id<>conversation.jsonl
   twarc2 csv --output-columns conversation_id,text sampleoutput_id<>conversation.jsonl sampleoutput_id<>conversation.csv


5、use concate to match