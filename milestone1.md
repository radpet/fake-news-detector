# Abstract
  The purpose of this document is to introduce the reader to our current research and development state
for fake news detection ana anylisis.


# Introduction

We are on the opinion that the task of detecting fake news is complex even for humans. That is why we decided to approach
the problem not by automatically detecting whether an article is fake or not but to analize the content and provide
meaningful insights such as sentiment, category, summary and many others.


# Stance Detection

## Motivation
Our main motives towards implementing a stance detection solution is pretty well said in the introduction of the [http://www.fakenewschallenge.org/](http://www.fakenewschallenge.org/) competion
> In particular, a good stance detection solution would allow a human fact checker to enter a claim or headline and instantly retrieve the top articles that agree, disagree or discuss the claim/headline in question. 
> They could then look at the arguments for and against the claim, and use their human judgment and reasoning skills to assess the validity of the claim in question. 
>Such a tool would enable human fact checkers to be fast and effective.

Our goal is not only to output whether the headline agree, disagree or discuss the main content but to find those phrases
in the text that contribute most to the result.

## Implementation
