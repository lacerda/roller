# roller

Roller is a movie recommendation system that understands movies as emotional rollercoasters.

## Motivation

I developed the concept of movie recommendations based on similarity of emotions over a timeline back in 2016.
The results weren't great due to many reasons, but the idea showed promise. I would like to try again with better tools and data.

## Current state

This proof of concept has a few shortcomings:
- Data is comprised of only 1000 movie scripts. These contain inconsistent structures, such as scene descriptions, character dialogue demarcations, dialogue only, official and fan-written.
- Emotion annotation depends on EmoLex. This is a dataset sourced from Mechanical Turk with many errors and does not take context into account.
- Scenes are not clearly demarcated. Each datum is simply an array of frequencies of each emotion in a section, where sections are partitions of text based on token length.

Moreover, this was developed in python 2.7, for which one of the dependencies (py_lex) is in a broken state.
