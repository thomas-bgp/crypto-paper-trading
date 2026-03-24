# VertoxQuant

**Data:** 2024-12-10T01:31:20.914Z
**URL:** https://www.vertoxquant.com/p/2026-quant-crash-course

---

Most quant roadmaps are useless.

They dump every book and paper the author has heard of (but never read). You’re left with a 50-item reading list and no idea where to start.

This roadmap is different. Every resource here has been tested. Every month builds on the last. And it contains just what you actually need to become a quant researcher.

VertoxQuant is a reader-supported publication. To receive new posts and support my work, consider becoming a free or paid subscriber.

Subscribe
Day 1: Python

As a quantitative researcher, 99% of your time will be spent with Python. You don’t need to be proficient in the language, but you should be able to code up any ideas that come to mind.

If you have no coding experience whatsoever, you can honestly just get started with any “Python for Beginners” YouTube video like this:

By itself, Python isn’t all that powerful for research. The reason it’s still the Nr 1 choice for most researchers is the libraries.

You will see the following 3 libraries in pretty much every project:

numpy (Fast library for working with arrays and matrices)

pandas (Data manipulation and analysis, Dataframe data structure)

matplotlib (Plotting)

Other than that, you still have many other libraries that make your life easier, like sklearn, scipy, polars, statsmodels, and cvxpy.

Your main goal when doing research should be to leverage libraries as much as you can to get results as quickly as possible and to move on to the next problem.

Don’t spend too much time in this stage. You can spend a day coding up a few small applications to get a basic understanding of the language, but other than that, you’ll learn the language by doing real quant projects.

Week 1 and 2: Calculus and Linear Algebra

You don’t need to be a PhD-level math genius to be doing quant (unless you want to be doing complex derivatives pricing, but we can learn that later).

A solid understanding on first semester college level Calculus and Linear Algebra will get you really far. There are 2 ways to learn this:

Go to university.

Read the following 2 articles that summarize everything in an intuitive way:

Math for Quant Finance (Calculus)
VERTOX
·
22 DE DEZEMBRO DE 2023

I get a lot of people asking what math they should learn for quant finance so I’m gonna summarize all of the most common and useful math that I use in quant finance in a series of articles!

Read full story
Math for Quant Finance (Linear Algebra)
VERTOX
·
30 DE DEZEMBRO DE 2023

Linear Algebra covers linear equations, linear maps, matrices and vectors.

Read full story

Universities heavily focus on proofs, as that’s what you need to discover (or invent?) new math. As a quant, I’ve never used the proof-writing skills that I learned in university before. Proofs can absolutely help with intuitive understanding of math, so if you want to gain a deeper understanding, try proving some things!

The book “How to Solve It” by George Polya teaches you a mental framework for how to approach proofs.

Just like with coding, you learn math by using it. That can either mean doing quant projects that involve a certain type of math or doing practice problems.

One of the best resources out there that I can recommend for both learning math and solving practice problems is Paul’s Online Notes:

https://tutorial.math.lamar.edu/

Week 3 and 4: Statistics

With the basic mathematical prerequisites down, you can move on to stuff you will apply more directly when doing research: Statistics!

There are 2 main branches of statistics:

Descriptive Statistics

This is all about summarizing data. Mean, Median, Variance, Volatility, Skewness, Correlation, Histograms, Boxplots, etc.

Those are all things you hear on a daily basis.

Inferential Statistics

This is all about inferring things. You are trying to predict the behavior of a non-observed set of information or generalize about a larger population using data from a sample. Things like linear regression, time series analysis, decision trees, and statistical tests are all part of inferential statistics.

The Organic Chemistry Tutor and StatQuest both have fantastic playlists on statistics:

https://www.youtube.com/playlist?list=PL0o_zxa4K1BVsziIRdfv4Hl4UIqDZhXWV

https://www.youtube.com/playlist?list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9

And again: Don’t just watch those videos blindly and try to remember everything, but actually try to apply what you’ve learned.
Just found out about histograms, boxplots, and other visualization techniques? Try to visualize some real financial data to build intuition!

Month 2: Machine Learning

Machine learning isn’t all about huge deep neural networks and LLMs. If I were to define it, I’d say it is “all about algorithms that learn patterns from data in order to make predictions or decisions, without being explicitly programmed for each task.”

You can again categorize types of machine learning models.

Supervised learning

You work with labeled data.

By far the most important models of this type are regression models:

Linear Regression

Logistic Regression

Polynomial Regression

Ridge Regression

Lasso Regression

Quantile Regression

Those are the machine learning models you’ll be working with most of the time, so you should get familiar with them.

https://www.geeksforgeeks.org/machine-learning/types-of-regression-techniques/

Another very powerful type of model is tree-based models, like:

Decision Trees

Random Forests

Gradient Boosted Trees (XGBoost, LightGBM)

You often see Gradient Boosted Trees winning Kaggle competitions.

Unsupervised Learning

No labels; you discover structure.

Examples:

Clustering (Grouping similar data)

Dimensionality Reduction (Finding structure in data)

Outlier Detection

Density Estimation

Clustering Algorithms - An in-depth view
VERTOX
·
8 DE DEZEMBRO DE 2023

In the previous article, we looked at latency arbitrage / lead-lag.

Read full story
Reinforcement Learning

Learning via interaction and feedback.

This is pretty niche, and typically other types of models do better anyway, so I wouldn’t recommend you to learn this if you are starting out.

StatQuest has a video on pretty much every topic I mentioned here, and more:

Most importantly: If your features and targets are garbage, no amount of model tuning will help you. Keep this in mind when working on projects!

Month 3: Optimization

Your chef comes up to you and tells you he wants a portfolio with 10% expected yearly return and volatility at most 8%. How do you do that? The answer: Optimization!

Optimization is an absolutely huge field, including:

Convex Optimization

Linear Programming

Semidefinite Programming

My favorite introductory book, which I would read myself if I had to start over again, is “Optimization Methods in Finance - Second Edition”.

This will also be your first time dealing with finance specifically (This goes to show how much of quant isn’t about finance, but about math and science.)

For a deep dive into Convex Optimization (which you don’t need at that level!), read “Convex Optimization” by Boyd and Vandenberghe.

Month 4: Portfolio Optimization and Management

You will already learn some portfolio optimization in the book we mentioned in the last section, but what I really like about “Robust Portfolio Optimization and Management (Frank J. Fabozzi Series)” is that it tells you about all the practically important techniques you need to know to create robust portfolios.

The truth is that if you just apply the portfolio optimization techniques you’ve learned about so far blindly to your sample returns and covariance, your portfolio weights will mostly be determined by noise.

Month 5: Numerical Methods

Most problems can’t be solved analytically, or it’s very cumbersome to do so. This is where numerical methods, which are incredibly powerful, come in!

Root Finding

The goal of root-finding algorithms is to figure out where a function is equal to 0. If you apply this to the derivative of a function, you can find its maxima and minima.

This is the only playlist you need:

https://www.youtube.com/playlist?list=PLb0Tx2oJWuYIpNE23qYHGQD42TIR3ThNz

Gradient Descent

The goal here is to find minimums / maximums of functions. There are many algorithms that are derived from basic gradient descent, so you should learn about it first:

Integration

Those are used to approximate integrals. Sometimes integrals are difficult or even impossible to solve analytically:

Differentiation

If we can integrate numerically, then we can also differentiate numerically:

Interpolation and Approximation

This is all about curve fitting and smoothing. The theory here can go really deep. This video explains the most important techniques:

Linear Algebra

Whenever you want to solve a linear system (Ax=b), compute eigenvalues and eigenvectors, or perform a matrix factorization, you need numerically stable algorithms. The following playlist covers all of those topics:


Those two playlists can be used as a lookup for anything about numerical methods:


https://www.youtube.com/playlist?list=PLDea8VeK4MUTppAXQzHBNz3KiyEd9SQms

https://www.youtube.com/playlist?list=PLkZjai-2Jcxn35XnijUtqqEg0Wi5Sn8ab

Month 6: Derivatives and Pricing

You can’t talk about derivatives and pricing without mentioning the two most iconic books:

“Options, Futures, and Other Derivatives” by John Hull

“Option Volatility and Pricing” by Natenberg.

The former is about all derivatives, while the latter covers options specifically.

You shouldn’t read them front-to-end, though, but look up what you want to become more familiar with.

QuantStart is a good start (pun not intended) to get familiar with the basics:


https://www.quantstart.com/articles/derivatives-pricing-i-pricing-under-the-black-scholes-model/

https://www.quantstart.com/articles/derivatives-pricing-ii-volatility-is-rough/

https://www.quantstart.com/articles/derivatives-pricing-iii-models-driven-by-levy-processes/

Months 7 and 8: Risk Management

I’ve written a huge article on risk management basics:


A Full Guide to Risk Management
VERTOX AND MALIK
·
14 DE JUNHO DE 2025

This is gonna be by far my biggest article, with the notebook from the risk manager that this was created with containing 1400+ lines of code and tons of visualizations! Actual risk management goes far beyond a stop loss order.

Read full story

If you care specifically about tail-risk, I recommend you learn about Extreme Value Theory. The following book is a perfect introduction:

“An Introduction to Statistical Modeling of Extreme Values” by Stuart Coles.

Volatility forecasting is naturally also very important for risk management:

Volatility Forecasting from High-Frequency Quotes
VERTOX
·
4 DE JAN.

Happy New Year, dear reader!

Read full story

There are many new topics you need to learn here, especially Extreme Value Theory, which is more challenging! Expect to spend some more time on this area. But trust me, it pays off!

Conclusion

Quant is a huge topic, and you can learn about it for the rest of your life.
This article tells you what to learn step by step to start doing real, meaningful research.

There are, of course, many more topics not covered in this crash course, like market microstructure, time series analysis, data engineering, execution algorithms, and everything high frequency. With the things you learned using the crash course, you are more than ready to dive into those topics yourself, though!

And again, because it’s so important: Don’t just learn by reading resources and watching videos. Apply what you learn to real projects to build REAL understanding and intuition!

Btw: I have written 77+ articles on everything quant on this site!

VertoxQuant
Quantitative Research