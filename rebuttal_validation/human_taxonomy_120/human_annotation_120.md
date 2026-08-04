# Human C0-C8 Annotation — 120 Claims

Label each claim using Appendix B. Do not open `private_id_mapping.json` or any judge-result files before completing the annotations.

## Category definitions and examples

### C0 — Timeless definition or logical/mathematical truth

Use C0 when the claim is true by definition or meaning and does not depend on a historical state of the world.

- Example: “The Brier score is the squared error of a probabilistic forecast.”
- Example: “A probability describes how likely an event is.”

Do **not** use C0 for ordinary real-world facts that merely happen to be stable; those are C1.

### C1 — Stable entity attribute

Use C1 for a real-world attribute that was established before the cutoff and is treated as stable through the relevant period.

- Example: “Paris is the capital of France.”
- Example: “The U.S. Supreme Court is composed of nine justices.”

If the attribute is explicitly indexed to a time, can change, or describes a temporary role/status, use C2 or a more specific temporal category.

### C2 — Time-indexed state, role, condition, or trend

Use C2 as the residual category for temporal facts that describe what was true during a period but are not a measurement, comparison, discrete event, or explicit publication claim.

- Example: “As of 2018, Angela Merkel served as Chancellor of Germany.”
- Example: “As of 2019, the United Kingdom remained an EU member.”

An “as of” phrase does not automatically imply C2: a numeric value is C3, and an explicit event is C5.

### C3 — Quantitative measurement

Use C3 when the claim states a measured numeric value, such as revenue, price, rate, count, score, salary, or percentage.

- Example: “Apple reported fiscal-2018 revenue of $265.6 billion.”
- Example: “The U.S. unemployment rate was 3.5% in September 2019.”

Classify by when the measurement became public, not merely the period being measured. If the claim explicitly attributes the value to a named report/publication, C6 has higher precedence.

### C4 — Comparative or relational claim

Use C4 when the claim compares two or more entities, values, periods, or states. Verification requires all compared facts.

- Example: “Apple’s fiscal-2018 revenue exceeded Amazon’s 2018 net sales.”
- Example: “The April-2020 unemployment rate was higher than the September-2019 rate.”

Use the latest availability date among the facts needed for the comparison.

### C5 — Discrete event

Use C5 when the claim states that a specific event occurred, such as an announcement, signing, election, launch, decision, filing, enactment, or appointment.

- Example: “The United Kingdom voted to leave the EU on June 23, 2016.”
- Example: “Microsoft won the JEDI cloud contract on October 25, 2019.”

If the event is the exact outcome being predicted, use C7 instead.

### C6 — Explicit publication or source claim

Use C6 when the claim explicitly attributes information to a dated article, report, filing, court opinion, press release, database, or other source.

- Example: “Apple’s November 1, 2018 earnings release reported $265.6 billion in revenue.”
- Example: “The October 4, 2019 BLS report stated that unemployment was 3.5%.”

C6 takes precedence over C3/C4/C5 because the claim’s provenance is explicitly tied to the publication.

### C7 — Target outcome

Use C7 when the claim states or directly reveals the exact result that the forecasting question asks the model to predict. Always read the provided **Target** field.

- Example: If predicting Kevin Durant’s contract AAV: “Durant signed a four-year, $164 million contract.”
- Example: If predicting an election winner: “Candidate X won the election.”

A target outcome may also look like a measurement or event; C7 has the highest precedence.

### C8 — Post-outcome consequence

Use C8 when the claim describes an effect that exists only after the target outcome occurs.

- Example: “The election result initiated the presidential transition.”
- Example: “The signing pushed the team above the luxury-tax threshold.”

If the claim states the target outcome itself, use C7; use C8 only for downstream consequences.

## Precedence and boundary rules

Apply the first matching category in this order:

**C7 > C8 > C6 > C3 > C4 > C5 > C2 > C1 > C0**

Important:

1. Classify the claim’s semantic type—not whether it is true, false, safe, or leaked.
2. Use the **Target** field to distinguish C7/C8 from ordinary events.
3. When multiple categories apply, always select the highest-precedence category.
4. If uncertain between adjacent categories, add a short note explaining the ambiguity.

Enter labels in `human_labels.csv`, or respond as:

```text
H001 | C0
H002 | C1
...
```

Confidence and notes are optional except for ambiguous cases.

## Claims

### H001

- Claim: Brier score is the squared error of a probabilistic forecast.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://en.wikipedia.org/wiki/Brier_score

### H002

- Claim: Gross domestic product measures the monetary value of final goods and services.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://en.wikipedia.org/wiki/Gross_domestic_product

### H003

- Claim: Spearman correlation measures rank correlation.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient

### H004

- Claim: A median separates the higher half from the lower half of a dataset.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://en.wikipedia.org/wiki/Median

### H005

- Claim: Variance measures dispersion around a mean.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://en.wikipedia.org/wiki/Variance

### H006

- Claim: Regression estimates relationships between a dependent variable and explanatory variables.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://en.wikipedia.org/wiki/Regression_analysis

### H007

- Claim: Binary classification assigns observations to one of two classes.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://en.wikipedia.org/wiki/Binary_classification

### H008

- Claim: A probability is a numerical description of how likely an event is.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://en.wikipedia.org/wiki/Probability

### H009

- Claim: Inflation is a general increase in prices.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://en.wikipedia.org/wiki/Inflation

### H010

- Claim: A confidence interval is a range of estimates for an unknown parameter.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://en.wikipedia.org/wiki/Confidence_interval

### H011

- Claim: Paris is the capital of France.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://en.wikipedia.org/wiki/Paris

### H012

- Claim: Canberra is the capital of Australia.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://en.wikipedia.org/wiki/Canberra

### H013

- Claim: The chemical symbol for gold is Au.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://en.wikipedia.org/wiki/Gold

### H014

- Claim: The Earth orbits the Sun.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://en.wikipedia.org/wiki/Earth%27s_orbit

### H015

- Claim: The Pacific Ocean is Earth's largest ocean.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://en.wikipedia.org/wiki/Pacific_Ocean

### H016

- Claim: Mount Everest is Earth's highest mountain above sea level.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://en.wikipedia.org/wiki/Mount_Everest

### H017

- Claim: The United Nations headquarters is in New York City.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://en.wikipedia.org/wiki/Headquarters_of_the_United_Nations

### H018

- Claim: The U.S. Supreme Court is composed of nine justices.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://www.supremecourt.gov/about/biographies.aspx

### H019

- Claim: The euro is the official currency of the euro area.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://en.wikipedia.org/wiki/Euro

### H020

- Claim: The International Court of Justice is based in The Hague.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://www.icj-cij.org/home

### H021

- Claim: As of 2018, Angela Merkel served as Chancellor of Germany.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://en.wikipedia.org/wiki/Angela_Merkel

### H022

- Claim: As of 2019, Theresa May served as Prime Minister of the United Kingdom.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://en.wikipedia.org/wiki/Theresa_May

### H023

- Claim: As of 2019, the United Kingdom remained a member of the European Union.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://en.wikipedia.org/wiki/Brexit

### H024

- Claim: As of 2018, Jeff Bezos was chief executive officer of Amazon.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://en.wikipedia.org/wiki/Jeff_Bezos

### H025

- Claim: As of 2019, the federal funds target range was below 2 percent.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm

### H026

- Claim: As of 2021, Joe Biden served as President of the United States.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://www.whitehouse.gov/about-the-white-house/presidents/joseph-r-biden-jr/

### H027

- Claim: As of 2022, Rishi Sunak served as Prime Minister of the United Kingdom.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://www.gov.uk/government/history/past-prime-ministers/rishi-sunak

### H028

- Claim: As of 2023, Twitter operated under the name X.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://en.wikipedia.org/wiki/Twitter,_Inc.

### H029

- Claim: As of 2021, the United Kingdom was no longer a member of the European Union.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://en.wikipedia.org/wiki/Brexit

### H030

- Claim: As of 2022, Elon Musk controlled Twitter.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://en.wikipedia.org/wiki/Acquisition_of_Twitter_by_Elon_Musk

### H031

- Claim: Apple reported fiscal 2018 revenue of $265.6 billion.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://www.apple.com/newsroom/2018/11/apple-reports-fourth-quarter-results/

### H032

- Claim: Microsoft reported fiscal 2019 revenue of $125.8 billion.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://www.microsoft.com/investor/reports/ar19/index.html

### H033

- Claim: The U.S. unemployment rate was 3.5 percent in September 2019.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://www.bls.gov/news.release/archives/empsit_10042019.htm

### H034

- Claim: Amazon reported 2018 net sales of $232.9 billion.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://ir.aboutamazon.com/news-release/news-release-details/2019/Amazoncom-Announces-Fourth-Quarter-Sales-up-20-to-724-Billion/default.aspx

### H035

- Claim: Alphabet reported 2019 revenue of $161.9 billion.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://abc.xyz/investor/news/news-details/2020/Alphabet-Announces-Fourth-Quarter-and-Fiscal-Year-2019-Results/

### H036

- Claim: Apple reported fiscal 2020 revenue of $274.5 billion.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://www.apple.com/newsroom/2020/10/apple-reports-fourth-quarter-results/

### H037

- Claim: The U.S. unemployment rate was 14.7 percent in April 2020.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://www.bls.gov/news.release/archives/empsit_05082020.htm

### H038

- Claim: Microsoft reported fiscal 2021 revenue of $168.1 billion.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://www.microsoft.com/en-us/Investor/earnings/FY-2021-Q4/press-release-webcast

### H039

- Claim: Meta reported 2022 revenue of $116.6 billion.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://investor.atmeta.com/investor-news/press-release-details/2023/Meta-Reports-Fourth-Quarter-and-Full-Year-2022-Results/default.aspx

### H040

- Claim: Apple reported fiscal 2023 revenue of $383.3 billion.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://www.apple.com/newsroom/2023/11/apple-reports-fourth-quarter-results/

### H041

- Claim: In 2018, Apple's revenue exceeded Microsoft's revenue.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://www.microsoft.com/investor/reports/ar19/index.html

### H042

- Claim: In September 2019, the U.S. unemployment rate was lower than in September 2018.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://www.bls.gov/news.release/archives/empsit_10042019.htm

### H043

- Claim: Apple's fiscal 2018 revenue exceeded Amazon's 2018 net sales.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://ir.aboutamazon.com/news-release/news-release-details/2019/Amazoncom-Announces-Fourth-Quarter-Sales-up-20-to-724-Billion/default.aspx

### H044

- Claim: Microsoft's fiscal 2019 revenue was lower than Alphabet's 2019 revenue.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://abc.xyz/investor/news/news-details/2020/Alphabet-Announces-Fourth-Quarter-and-Fiscal-Year-2019-Results/

### H045

- Claim: The April 2020 U.S. unemployment rate was higher than the September 2019 rate.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://www.bls.gov/news.release/archives/empsit_05082020.htm

### H046

- Claim: Apple's fiscal 2020 revenue exceeded its fiscal 2018 revenue.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://www.apple.com/newsroom/2020/10/apple-reports-fourth-quarter-results/

### H047

- Claim: Microsoft's fiscal 2021 revenue exceeded Alphabet's 2019 revenue.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://www.microsoft.com/en-us/Investor/earnings/FY-2021-Q4/press-release-webcast

### H048

- Claim: Meta's 2022 revenue was lower than Microsoft's fiscal 2021 revenue.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://investor.atmeta.com/investor-news/press-release-details/2023/Meta-Reports-Fourth-Quarter-and-Full-Year-2022-Results/default.aspx

### H049

- Claim: Apple's fiscal 2023 revenue exceeded its fiscal 2020 revenue.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://www.apple.com/newsroom/2023/11/apple-reports-fourth-quarter-results/

### H050

- Claim: The September 2019 unemployment rate was lower than the April 2020 rate.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://www.bls.gov/news.release/archives/empsit_05082020.htm

### H051

- Claim: The United Kingdom voted to leave the European Union on June 23, 2016.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://en.wikipedia.org/wiki/2016_United_Kingdom_European_Union_membership_referendum

### H052

- Claim: The United States inaugurated Donald Trump as president on January 20, 2017.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://en.wikipedia.org/wiki/Inauguration_of_Donald_Trump

### H053

- Claim: Apple released the iPhone X on November 3, 2017.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://www.apple.com/newsroom/2017/09/the-future-is-here-iphone-x/

### H054

- Claim: Microsoft won the JEDI cloud contract on October 25, 2019.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://www.defense.gov/News/Releases/Release/Article/1999635/department-of-defense-awards-jedi-cloud-contract/

### H055

- Claim: The World Health Organization was informed of pneumonia cases in Wuhan on December 31, 2019.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://www.who.int/emergencies/diseases/novel-coronavirus-2019/interactive-timeline

### H056

- Claim: The World Health Organization declared COVID-19 a pandemic on March 11, 2020.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://www.who.int/director-general/speeches/detail/who-director-general-s-opening-remarks-at-the-media-briefing-on-covid-19---11-march-2020

### H057

- Claim: The United Kingdom left the European Union on January 31, 2020.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://en.wikipedia.org/wiki/Brexit

### H058

- Claim: Joe Biden was inaugurated as U.S. president on January 20, 2021.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://en.wikipedia.org/wiki/Inauguration_of_Joe_Biden

### H059

- Claim: Russia launched a full-scale invasion of Ukraine on February 24, 2022.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://en.wikipedia.org/wiki/Russian_invasion_of_Ukraine

### H060

- Claim: Elon Musk completed the acquisition of Twitter on October 27, 2022.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://en.wikipedia.org/wiki/Acquisition_of_Twitter_by_Elon_Musk

### H061

- Claim: Apple's November 1, 2018 earnings release reported fiscal 2018 revenue of $265.6 billion.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://www.apple.com/newsroom/2018/11/apple-reports-fourth-quarter-results/

### H062

- Claim: The October 4, 2019 BLS employment report stated that unemployment was 3.5 percent.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://www.bls.gov/news.release/archives/empsit_10042019.htm

### H063

- Claim: Microsoft's August 1, 2019 annual report stated fiscal 2019 revenue of $125.8 billion.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://www.microsoft.com/investor/reports/ar19/index.html

### H064

- Claim: The WHO's December 31, 2019 timeline entry recorded pneumonia cases in Wuhan.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://www.who.int/emergencies/diseases/novel-coronavirus-2019/interactive-timeline

### H065

- Claim: Alphabet's February 3, 2020 release reported 2019 revenue of $161.9 billion.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://abc.xyz/investor/news/news-details/2020/Alphabet-Announces-Fourth-Quarter-and-Fiscal-Year-2019-Results/

### H066

- Claim: The WHO's March 11, 2020 briefing characterized COVID-19 as a pandemic.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://www.who.int/director-general/speeches/detail/who-director-general-s-opening-remarks-at-the-media-briefing-on-covid-19---11-march-2020

### H067

- Claim: Apple's October 29, 2020 earnings release reported fiscal 2020 revenue of $274.5 billion.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://www.apple.com/newsroom/2020/10/apple-reports-fourth-quarter-results/

### H068

- Claim: Microsoft's July 27, 2021 release reported fiscal 2021 revenue of $168.1 billion.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://www.microsoft.com/en-us/Investor/earnings/FY-2021-Q4/press-release-webcast

### H069

- Claim: Meta's February 1, 2023 release reported 2022 revenue of $116.6 billion.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://investor.atmeta.com/investor-news/press-release-details/2023/Meta-Reports-Fourth-Quarter-and-Full-Year-2022-Results/default.aspx

### H070

- Claim: Apple's November 2, 2023 earnings release reported fiscal 2023 revenue of $383.3 billion.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://www.apple.com/newsroom/2023/11/apple-reports-fourth-quarter-results/

### H071

- Claim: The United Kingdom voted to leave the European Union.
- Cutoff: 2015-01-01
- Target: whether the United Kingdom would vote to leave the European Union
- Source: https://en.wikipedia.org/wiki/Main_Page

### H072

- Claim: Donald Trump won the 2016 U.S. presidential election.
- Cutoff: 2015-01-01
- Target: whether Donald Trump would win the 2016 U.S. presidential election
- Source: https://en.wikipedia.org/wiki/Main_Page

### H073

- Claim: Microsoft won the JEDI cloud contract.
- Cutoff: 2018-01-01
- Target: which company would win the JEDI cloud contract
- Source: https://en.wikipedia.org/wiki/Main_Page

### H074

- Claim: The United Kingdom left the European Union.
- Cutoff: 2019-01-01
- Target: whether the United Kingdom would leave the European Union
- Source: https://en.wikipedia.org/wiki/Main_Page

### H075

- Claim: The WHO declared COVID-19 a pandemic.
- Cutoff: 2019-01-01
- Target: whether the WHO would declare COVID-19 a pandemic
- Source: https://en.wikipedia.org/wiki/Main_Page

### H076

- Claim: Joe Biden won the 2020 U.S. presidential election.
- Cutoff: 2019-01-01
- Target: whether Joe Biden would win the 2020 U.S. presidential election
- Source: https://en.wikipedia.org/wiki/Main_Page

### H077

- Claim: Joe Biden was inaugurated as U.S. president.
- Cutoff: 2020-01-01
- Target: whether Joe Biden would be inaugurated as U.S. president
- Source: https://en.wikipedia.org/wiki/Main_Page

### H078

- Claim: Russia launched a full-scale invasion of Ukraine.
- Cutoff: 2021-01-01
- Target: whether Russia would launch a full-scale invasion of Ukraine
- Source: https://en.wikipedia.org/wiki/Main_Page

### H079

- Claim: Elon Musk completed the acquisition of Twitter.
- Cutoff: 2021-01-01
- Target: whether Elon Musk would complete the acquisition of Twitter
- Source: https://en.wikipedia.org/wiki/Main_Page

### H080

- Claim: Argentina won the 2022 FIFA World Cup.
- Cutoff: 2021-01-01
- Target: which country would win the 2022 FIFA World Cup
- Source: https://en.wikipedia.org/wiki/Main_Page

### H081

- Claim: The Brexit referendum result led David Cameron to announce his resignation.
- Cutoff: 2015-01-01
- Target: whether the United Kingdom would vote to leave the European Union
- Source: https://en.wikipedia.org/wiki/Main_Page

### H082

- Claim: Donald Trump's election victory initiated his presidential transition.
- Cutoff: 2015-01-01
- Target: whether Donald Trump would win the 2016 U.S. presidential election
- Source: https://en.wikipedia.org/wiki/Main_Page

### H083

- Claim: The JEDI award triggered a legal challenge by Amazon.
- Cutoff: 2018-01-01
- Target: which company would win the JEDI cloud contract
- Source: https://en.wikipedia.org/wiki/Main_Page

### H084

- Claim: The United Kingdom's departure began the post-Brexit transition period.
- Cutoff: 2019-01-01
- Target: whether the United Kingdom would leave the European Union
- Source: https://en.wikipedia.org/wiki/Main_Page

### H085

- Claim: The pandemic declaration led governments to intensify public-health restrictions.
- Cutoff: 2019-01-01
- Target: whether the WHO would declare COVID-19 a pandemic
- Source: https://en.wikipedia.org/wiki/Main_Page

### H086

- Claim: Joe Biden's election victory initiated his presidential transition.
- Cutoff: 2019-01-01
- Target: whether Joe Biden would win the 2020 U.S. presidential election
- Source: https://en.wikipedia.org/wiki/Main_Page

### H087

- Claim: Joe Biden's inauguration made Kamala Harris vice president.
- Cutoff: 2020-01-01
- Target: whether Joe Biden would be inaugurated as U.S. president
- Source: https://en.wikipedia.org/wiki/Main_Page

### H088

- Claim: Russia's invasion led to extensive international sanctions.
- Cutoff: 2021-01-01
- Target: whether Russia would launch a full-scale invasion of Ukraine
- Source: https://en.wikipedia.org/wiki/Main_Page

### H089

- Claim: Musk's Twitter acquisition made the company privately held.
- Cutoff: 2021-01-01
- Target: whether Elon Musk would complete the acquisition of Twitter
- Source: https://en.wikipedia.org/wiki/Main_Page

### H090

- Claim: Argentina's World Cup victory prompted championship celebrations in Buenos Aires.
- Cutoff: 2021-01-01
- Target: which country would win the 2022 FIFA World Cup
- Source: https://en.wikipedia.org/wiki/Main_Page

### H091

- Claim: Apple's November 1, 2018 earnings release reported fiscal 2018 revenue of $265.6 billion.
- Cutoff: 2017-01-01
- Target: not applicable
- Source: https://www.apple.com/newsroom/2018/11/apple-reports-fourth-quarter-results/

### H092

- Claim: Apple's November 1, 2018 earnings release reported fiscal 2018 revenue of $265.6 billion.
- Cutoff: 2019-12-31
- Target: not applicable
- Source: https://www.apple.com/newsroom/2018/11/apple-reports-fourth-quarter-results/

### H093

- Claim: The October 4, 2019 BLS employment report stated that unemployment was 3.5 percent.
- Cutoff: 2018-01-01
- Target: not applicable
- Source: https://www.bls.gov/news.release/archives/empsit_10042019.htm

### H094

- Claim: The October 4, 2019 BLS employment report stated that unemployment was 3.5 percent.
- Cutoff: 2020-12-31
- Target: not applicable
- Source: https://www.bls.gov/news.release/archives/empsit_10042019.htm

### H095

- Claim: Microsoft's August 1, 2019 annual report stated fiscal 2019 revenue of $125.8 billion.
- Cutoff: 2018-01-01
- Target: not applicable
- Source: https://www.microsoft.com/investor/reports/ar19/index.html

### H096

- Claim: Microsoft's August 1, 2019 annual report stated fiscal 2019 revenue of $125.8 billion.
- Cutoff: 2020-12-31
- Target: not applicable
- Source: https://www.microsoft.com/investor/reports/ar19/index.html

### H097

- Claim: The WHO's December 31, 2019 timeline entry recorded pneumonia cases in Wuhan.
- Cutoff: 2018-01-01
- Target: not applicable
- Source: https://www.who.int/emergencies/diseases/novel-coronavirus-2019/interactive-timeline

### H098

- Claim: The WHO's December 31, 2019 timeline entry recorded pneumonia cases in Wuhan.
- Cutoff: 2020-12-31
- Target: not applicable
- Source: https://www.who.int/emergencies/diseases/novel-coronavirus-2019/interactive-timeline

### H099

- Claim: Alphabet's February 3, 2020 release reported 2019 revenue of $161.9 billion.
- Cutoff: 2019-01-01
- Target: not applicable
- Source: https://abc.xyz/investor/news/news-details/2020/Alphabet-Announces-Fourth-Quarter-and-Fiscal-Year-2019-Results/

### H100

- Claim: Alphabet's February 3, 2020 release reported 2019 revenue of $161.9 billion.
- Cutoff: 2021-12-31
- Target: not applicable
- Source: https://abc.xyz/investor/news/news-details/2020/Alphabet-Announces-Fourth-Quarter-and-Fiscal-Year-2019-Results/

### H101

- Claim: The WHO's March 11, 2020 briefing characterized COVID-19 as a pandemic.
- Cutoff: 2019-01-01
- Target: not applicable
- Source: https://www.who.int/director-general/speeches/detail/who-director-general-s-opening-remarks-at-the-media-briefing-on-covid-19---11-march-2020

### H102

- Claim: The WHO's March 11, 2020 briefing characterized COVID-19 as a pandemic.
- Cutoff: 2021-12-31
- Target: not applicable
- Source: https://www.who.int/director-general/speeches/detail/who-director-general-s-opening-remarks-at-the-media-briefing-on-covid-19---11-march-2020

### H103

- Claim: Apple's October 29, 2020 earnings release reported fiscal 2020 revenue of $274.5 billion.
- Cutoff: 2019-01-01
- Target: not applicable
- Source: https://www.apple.com/newsroom/2020/10/apple-reports-fourth-quarter-results/

### H104

- Claim: Apple's October 29, 2020 earnings release reported fiscal 2020 revenue of $274.5 billion.
- Cutoff: 2021-12-31
- Target: not applicable
- Source: https://www.apple.com/newsroom/2020/10/apple-reports-fourth-quarter-results/

### H105

- Claim: Microsoft's July 27, 2021 release reported fiscal 2021 revenue of $168.1 billion.
- Cutoff: 2020-01-01
- Target: not applicable
- Source: https://www.microsoft.com/en-us/Investor/earnings/FY-2021-Q4/press-release-webcast

### H106

- Claim: Microsoft's July 27, 2021 release reported fiscal 2021 revenue of $168.1 billion.
- Cutoff: 2022-12-31
- Target: not applicable
- Source: https://www.microsoft.com/en-us/Investor/earnings/FY-2021-Q4/press-release-webcast

### H107

- Claim: Meta's February 1, 2023 release reported 2022 revenue of $116.6 billion.
- Cutoff: 2022-01-01
- Target: not applicable
- Source: https://investor.atmeta.com/investor-news/press-release-details/2023/Meta-Reports-Fourth-Quarter-and-Full-Year-2022-Results/default.aspx

### H108

- Claim: Meta's February 1, 2023 release reported 2022 revenue of $116.6 billion.
- Cutoff: 2024-12-31
- Target: not applicable
- Source: https://investor.atmeta.com/investor-news/press-release-details/2023/Meta-Reports-Fourth-Quarter-and-Full-Year-2022-Results/default.aspx

### H109

- Claim: Apple's November 2, 2023 earnings release reported fiscal 2023 revenue of $383.3 billion.
- Cutoff: 2022-01-01
- Target: not applicable
- Source: https://www.apple.com/newsroom/2023/11/apple-reports-fourth-quarter-results/

### H110

- Claim: Apple's November 2, 2023 earnings release reported fiscal 2023 revenue of $383.3 billion.
- Cutoff: 2024-12-31
- Target: not applicable
- Source: https://www.apple.com/newsroom/2023/11/apple-reports-fourth-quarter-results/

### H111

- Claim: Apple's November 1, 2018 earnings release reported fiscal 2018 revenue of $265.6 billion.
- Cutoff: 2016-06-30
- Target: not applicable
- Source: https://www.apple.com/newsroom/2018/11/apple-reports-fourth-quarter-results/

### H112

- Claim: Apple's November 1, 2018 earnings release reported fiscal 2018 revenue of $265.6 billion.
- Cutoff: 2020-06-30
- Target: not applicable
- Source: https://www.apple.com/newsroom/2018/11/apple-reports-fourth-quarter-results/

### H113

- Claim: The October 4, 2019 BLS employment report stated that unemployment was 3.5 percent.
- Cutoff: 2017-06-30
- Target: not applicable
- Source: https://www.bls.gov/news.release/archives/empsit_10042019.htm

### H114

- Claim: The October 4, 2019 BLS employment report stated that unemployment was 3.5 percent.
- Cutoff: 2021-06-30
- Target: not applicable
- Source: https://www.bls.gov/news.release/archives/empsit_10042019.htm

### H115

- Claim: Microsoft's August 1, 2019 annual report stated fiscal 2019 revenue of $125.8 billion.
- Cutoff: 2017-06-30
- Target: not applicable
- Source: https://www.microsoft.com/investor/reports/ar19/index.html

### H116

- Claim: Microsoft's August 1, 2019 annual report stated fiscal 2019 revenue of $125.8 billion.
- Cutoff: 2021-06-30
- Target: not applicable
- Source: https://www.microsoft.com/investor/reports/ar19/index.html

### H117

- Claim: The WHO's December 31, 2019 timeline entry recorded pneumonia cases in Wuhan.
- Cutoff: 2017-06-30
- Target: not applicable
- Source: https://www.who.int/emergencies/diseases/novel-coronavirus-2019/interactive-timeline

### H118

- Claim: The WHO's December 31, 2019 timeline entry recorded pneumonia cases in Wuhan.
- Cutoff: 2021-06-30
- Target: not applicable
- Source: https://www.who.int/emergencies/diseases/novel-coronavirus-2019/interactive-timeline

### H119

- Claim: Alphabet's February 3, 2020 release reported 2019 revenue of $161.9 billion.
- Cutoff: 2018-06-30
- Target: not applicable
- Source: https://abc.xyz/investor/news/news-details/2020/Alphabet-Announces-Fourth-Quarter-and-Fiscal-Year-2019-Results/

### H120

- Claim: Alphabet's February 3, 2020 release reported 2019 revenue of $161.9 billion.
- Cutoff: 2022-06-30
- Target: not applicable
- Source: https://abc.xyz/investor/news/news-details/2020/Alphabet-Announces-Fourth-Quarter-and-Fiscal-Year-2019-Results/
