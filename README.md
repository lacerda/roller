# roller

Roller is a movie recommendation system that understands stories as an array of emotion frequencies.

## Motivation

I developed the concept of recommendations based on similarity of emotions over a timeline back in 2016 as a response to most movie recommendation systems being based on collaborative filtering, or based on theme and genre recommendations.

## Emotion based representations

Instead of looking superficially at words in a story, consider how they make us feel. There are many emotion annotation systems such as NRC EmoLex, WordNet-Affect, Affective Text, themselves based on studies by Eckman or Plutchik. By tracking the frequency of emotions throughout a story, an emotion-based representation is produced, allowing comparison with other stories as well as recommendations.

A proof-of-concept of this idea is available in this repository at ('Recommending stories based on emotional timelines.pdf')[https://github.com/lacerda/roller/blob/master/Recommending%20stories%20based%20on%20emotional%20timelines.pdf].
