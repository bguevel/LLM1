There are two projects that exist within the repo one is more made by Luca while the other is more made
by Ben and Cole.

Cole helped a significant amounnt with figuring out bugs in the training and embedding. Along with that he helped figure out the math so
we could get it pair programmed into code.

To operate the gpt that Ben and Cole made simply run uv sync in the highest level directory with the toml file
Then press run or use python console commands to run LLM3.py, the follow the prompts given in the terminal

******* If uv sync doesn't work it could be because the uv.lock file already exists and needs to be deleted ********


We chose to have a multiheaded attention head, and a word level tokenizer. Along with that we made a vocabulary that expands 
when it encounters new words, as we didn't want to use a built in library for the vocab, and we also wanted it to be able to learn new words on
the fly. This came with a whole host of challenges as the embedding depends on the size of the vocabulary so we had to on the fly
edit the embedding.

For the future we would look into byte encoding, and maybe a few other possible avenues.
