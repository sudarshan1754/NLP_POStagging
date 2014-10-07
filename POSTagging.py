###********************************************************************************###
  # __author__ = 'sid'                                                             #
  # This program is written as part of the Natural Language Processing Home Work 2 #
  # @copyright: Sudarshan Sudarshan (Sid)                                          #
###********************************************************************************###

import nltk
import math

# hw3_train


class pos_training:

    @staticmethod
    def tokenization(fpath):

        pos = {}
        no_of_tags = 0
        word_tag ={}
        transition = {}

        file_content = open(fpath)

        for line in file_content.readlines():
            tokens = nltk.WhitespaceTokenizer().tokenize(line)

            for index, token in enumerate(tokens):  # Create the dictionary

                # Increment the No_of_tags by 1
                no_of_tags += 1

                # Add the <word tag: count> to dictionary
                word = token.split("/")[0]
                tag = token.split("/")[1]
                if  word + " " + tag in word_tag:
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

        # print len(pos)
        # print len(transition)
        # print len(word_tag)

        token_results = [pos, transition, word_tag, no_of_tags]

        return token_results

    @staticmethod
    def calculate_probability(bigrams, pos):

        bigram_prob = {}
        for word, count in bigrams.items():
            bigram_prob[word] = count / float(pos[word.split(" ")[1]])

        return bigram_prob


if __name__ == "__main__":

    print "\n-------------------------Welcome-------------------------\n"

    while True:

        option = raw_input('1. Train the Model\n2. Test the Language Model on a file\n3. Exit\n\nEnter your choice:')

        if int(option) == 1:

            # get the training file name from the user
            trainfile = raw_input('Enter the training file:')
            lmpath = raw_input('Enter the LM file name: ')

            # get the training results
            train = pos_training()
            token_results = train.tokenization(trainfile)

            # to get the transition probabilities
            tran_results = train.calculate_probability(token_results[1], token_results[0])

            # to get the and observation probabilities
            obs_results = train.calculate_probability(token_results[2], token_results[0])

            # store the language model file
            LM_file = open(lmpath, "w")

            # to store pos
            LM_file.write("pos:\n")
            pos = token_results[0]
            for tag in pos:
                LM_file.write(str(tag) + "\t" + str(pos[tag]) + "\n")

            # to store transition probs
            LM_file.write("transition:\n")
            for tag_tag in tran_results:
                # LM_file.write(str(tag_tag) + "\t" + str(math.log(tran_results[tag_tag], 2)) + "\n")
                LM_file.write(str(tag_tag) + "\t" + str((tran_results[tag_tag])) + "\n")

            # to store observation probs
            LM_file.write("observation:\n")
            for word_tag in obs_results:
                # LM_file.write(str(word_tag) + "\t" + str(math.log(obs_results[word_tag], 2)) + "\n")
                LM_file.write(str(word_tag) + "\t" + str((obs_results[word_tag])) + "\n")

            LM_file.close()

        elif int(option) == 3:
            print "-------------------------Good Bye-------------------------"
            break

        else:
            print "------Your choice is not valid. Enter a valid choice!------\n"





