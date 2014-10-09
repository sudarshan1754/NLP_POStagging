###********************************************************************************###
  # __author__ = 'sid'                                                             #
  # This program is written as part of the Natural Language Processing Home Work 2 #
  # @copyright: Sudarshan Sudarshan (Sid)                                          #
###********************************************************************************###

import nltk
import math
import time


class pos_training:
    @staticmethod
    def tokenization(fpath):

        pos = {}
        no_of_tags = 0
        word_tag = {}
        transition = {}

        starttags = ["<s>/<s>"]  # Dummy start symbol
        endtags = ["<e>/<e>"]  # Dummy end symbol

        file_content = open(fpath)

        for line in file_content.readlines():
            tokens = starttags + nltk.WhitespaceTokenizer().tokenize(line) + endtags

            for index, token in enumerate(tokens):  # Create the dictionary

                # Increment the No_of_tags by 1
                no_of_tags += 1

                # Add the <word tag: count> to dictionary
                word = token.split("/")[0]
                tag = token.split("/")[1]
                if word + " " + tag in word_tag:
                    word_tag[word + " " + tag] += 1
                else:
                    word_tag[word + " " + tag] = 1

                # Add the pos occurrence to dictionary
                if tag in pos:
                    pos[tag] += 1
                else:
                    pos[tag] = 1

                # Get the transition tags
                if index < len(tokens) - 1:
                    tag1 = tokens[index].split("/")[1]
                    tag2 = tokens[index + 1].split("/")[1]
                    if (tag1 + " " + tag2) in transition:
                        transition[tag1 + " " + tag2] += 1
                    else:
                        transition[tag1 + " " + tag2] = 1

        token_results = [pos, transition, word_tag, no_of_tags]

        return token_results

    # Function to calculate the Unigram probability of tags
    @staticmethod
    def Unigram_Probability(pos, No_of_tags):

        # A dictionary to store the <unigram: probability>
        pos_probability = {}

        for word, count in pos.items():
            # Use MLE Estimation
            pos_probability[word] = (count / float(No_of_tags))

        return pos_probability

    # Function to calculate tag_tag probability
    @staticmethod
    def tagtag_probability(tagtag, pos, pos_prob):

        # For transition probability we must use the Interpolation smoothing technique
        alpha = 0.99
        beta = 1 - alpha

        tagtag_prob = {}
        for word, count in tagtag.items():
            tagtag_prob[word] = (alpha * (count / float(pos[word.split(" ")[1]]))) + (
                beta * float(pos_prob[word.split(" ")[0]]))

        return tagtag_prob

    # Function to calculate word_tag probability
    @staticmethod
    def wordtag_probability(wordtag, pos):

        wordtag_prob = {}
        for word, count in wordtag.items():
            wordtag_prob[word] = (count / float(pos[word.split(" ")[1]]))

        return wordtag_prob


if __name__ == "__main__":

    print "\n-------------------------Welcome-------------------------\n"

    while True:

        option = raw_input('1. Train the Model\n2. Test the Language Model on a file\n3. Exit\n\nEnter your choice:')

        if int(option) == 1:

            # get the training file name from the user
            trainfile = raw_input('Enter the training file:')
            lmpath = raw_input('Enter the LM file name: ')

            # Timer to get the execution time
            start_time = time.time()

            # get the training results
            train = pos_training()
            token_results = train.tokenization(trainfile)

            # get the ml of tags
            pos_prob = train.Unigram_Probability(token_results[0], token_results[3])

            # to get the transition probabilities
            tran_results = train.tagtag_probability(token_results[1], token_results[0], pos_prob)

            # to get the and observation probabilities
            obs_results = train.wordtag_probability(token_results[2], token_results[0])

            # store the language model file
            LM_file = open(lmpath, "w")

            # to store pos_prob
            LM_file.write("pos:\n")
            for tag in pos_prob:
                LM_file.write(str(tag) + "\t" + str(math.log(pos_prob[tag], 2)) + "\n")

            # to store transition probs
            LM_file.write("transition:\n")
            for tag_tag in tran_results:
                LM_file.write(str(tag_tag) + "\t" + str(math.log(tran_results[tag_tag], 2)) + "\n")

            # to store observation probs
            LM_file.write("observation:\n")
            for word_tag in obs_results:
                LM_file.write(str(word_tag) + "\t" + str(math.log(obs_results[word_tag], 2)) + "\n")

            LM_file.close()

            print "\n--- %s seconds ---\n" % (time.time() - start_time)

        elif int(option) == 3:
            print "-------------------------Good Bye-------------------------"
            break

        else:
            print "------Your choice is not valid. Enter a valid choice!------\n"





