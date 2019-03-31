#!/usr/bin/env python

# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def run_quickstart():
    # [START translate_quickstart]
    # Imports the Google Cloud client library
    from google.cloud import translate
    import codecs
    import sys

    # Instantiates a client
    translate_client = translate.Client()
    input_path = sys.argv[1]

    # The text to translate


    #text = u'Chris You are just stiring the pot and it is a political dig. How about Texas how were things handled there. PR was a totally different situation.  How did you help?'
# The target language
    target = ['ru', 'ar', 'zh-TW']

# Translates some text into Russian

    for tar_lang in target:
        with codecs.open(input_path+str(tar_lang), 'w', 'utf-8') as out_obj:
            with codecs.open(input_path, 'r', 'utf-8') as in_obj:
                for line in in_obj:    
                    translation = translate_client.translate(line.split('\t')[1], target_language=tar_lang, source_language='en' )

                # print(u'Text: {}'.format(each_text))
                # print(u'Translation: {}'.format(translation['translatedText']))

                    translation_back = translate_client.translate(translation['translatedText'], target_language='en')
                    out_obj.write("%s\t%s\t%s" %(line.split('\t')[0],translation_back['translatedText'], line.split('\t')[2]))
            # print(u'Translation: {}'.format(translation_back['translatedText']))
            # print('=============================================================')
        # [END translate_quickstart]


if __name__ == '__main__':
    run_quickstart()

