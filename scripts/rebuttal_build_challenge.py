"""Build a compact public-source C0-C8 and date-boundary challenge set.

The set is deliberately composed of clear cases. Labels are rule/source-derived,
not independent human annotations. It is intended for rebuttal robustness checks.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "rebuttal_validation" / "data" / "taxonomy_date_challenge.jsonl"
DEFAULT_CUTOFF = "2020-01-01"


def item(
    item_id: str,
    category: str,
    fact: str,
    source_date: str | None,
    source_url: str,
    *,
    cutoff: str = DEFAULT_CUTOFF,
    target: str = "",
    subset: str = "taxonomy",
) -> dict:
    if category in {"C0", "C1"}:
        tier, leaked = "safe", False
    elif category in {"C7", "C8"}:
        tier, leaked = "leaked", True
    else:
        tier = "verify"
        if source_date is None:
            raise ValueError(f"{item_id}: date-dependent claim requires source_date")
        leaked = source_date > cutoff
    return {
        "id": item_id,
        "subset": subset,
        "fact": fact,
        "declared_source_date": source_date,
        "gold_effective_source_date": source_date,
        "cutoff_date": cutoff,
        "task": "binary",
        "target_description": target,
        "gold_category": category,
        "gold_tier": tier,
        "gold_is_leaked": leaked,
        "source_url": source_url,
    }


def build_taxonomy_items() -> list[dict]:
    rows: list[dict] = []
    definitions = [
        ("Brier score is the squared error of a probabilistic forecast.", "https://en.wikipedia.org/wiki/Brier_score"),
        ("Gross domestic product measures the monetary value of final goods and services.", "https://en.wikipedia.org/wiki/Gross_domestic_product"),
        ("Spearman correlation measures rank correlation.", "https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient"),
        ("A median separates the higher half from the lower half of a dataset.", "https://en.wikipedia.org/wiki/Median"),
        ("Variance measures dispersion around a mean.", "https://en.wikipedia.org/wiki/Variance"),
        ("Regression estimates relationships between a dependent variable and explanatory variables.", "https://en.wikipedia.org/wiki/Regression_analysis"),
        ("Binary classification assigns observations to one of two classes.", "https://en.wikipedia.org/wiki/Binary_classification"),
        ("A probability is a numerical description of how likely an event is.", "https://en.wikipedia.org/wiki/Probability"),
        ("Inflation is a general increase in prices.", "https://en.wikipedia.org/wiki/Inflation"),
        ("A confidence interval is a range of estimates for an unknown parameter.", "https://en.wikipedia.org/wiki/Confidence_interval"),
    ]
    for idx, (fact, url) in enumerate(definitions, 1):
        rows.append(item(f"c0_{idx:02d}", "C0", fact, None, url))

    stable = [
        ("Paris is the capital of France.", "https://en.wikipedia.org/wiki/Paris"),
        ("Canberra is the capital of Australia.", "https://en.wikipedia.org/wiki/Canberra"),
        ("The chemical symbol for gold is Au.", "https://en.wikipedia.org/wiki/Gold"),
        ("The Earth orbits the Sun.", "https://en.wikipedia.org/wiki/Earth%27s_orbit"),
        ("The Pacific Ocean is Earth's largest ocean.", "https://en.wikipedia.org/wiki/Pacific_Ocean"),
        ("Mount Everest is Earth's highest mountain above sea level.", "https://en.wikipedia.org/wiki/Mount_Everest"),
        ("The United Nations headquarters is in New York City.", "https://en.wikipedia.org/wiki/Headquarters_of_the_United_Nations"),
        ("The U.S. Supreme Court is composed of nine justices.", "https://www.supremecourt.gov/about/biographies.aspx"),
        ("The euro is the official currency of the euro area.", "https://en.wikipedia.org/wiki/Euro"),
        ("The International Court of Justice is based in The Hague.", "https://www.icj-cij.org/home"),
    ]
    for idx, (fact, url) in enumerate(stable, 1):
        rows.append(item(f"c1_{idx:02d}", "C1", fact, None, url))

    c2 = [
        ("As of 2018, Angela Merkel served as Chancellor of Germany.", "2018-01-01", "https://en.wikipedia.org/wiki/Angela_Merkel"),
        ("As of 2019, Theresa May served as Prime Minister of the United Kingdom.", "2019-01-01", "https://en.wikipedia.org/wiki/Theresa_May"),
        ("As of 2019, the United Kingdom remained a member of the European Union.", "2019-01-01", "https://en.wikipedia.org/wiki/Brexit"),
        ("As of 2018, Jeff Bezos was chief executive officer of Amazon.", "2018-01-01", "https://en.wikipedia.org/wiki/Jeff_Bezos"),
        ("As of 2019, the federal funds target range was below 2 percent.", "2019-12-11", "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"),
        ("As of 2021, Joe Biden served as President of the United States.", "2021-01-20", "https://www.whitehouse.gov/about-the-white-house/presidents/joseph-r-biden-jr/"),
        ("As of 2022, Rishi Sunak served as Prime Minister of the United Kingdom.", "2022-10-25", "https://www.gov.uk/government/history/past-prime-ministers/rishi-sunak"),
        ("As of 2023, Twitter operated under the name X.", "2023-07-24", "https://en.wikipedia.org/wiki/Twitter,_Inc."),
        ("As of 2021, the United Kingdom was no longer a member of the European Union.", "2021-01-01", "https://en.wikipedia.org/wiki/Brexit"),
        ("As of 2022, Elon Musk controlled Twitter.", "2022-10-27", "https://en.wikipedia.org/wiki/Acquisition_of_Twitter_by_Elon_Musk"),
    ]
    for idx, (fact, date, url) in enumerate(c2, 1):
        rows.append(item(f"c2_{idx:02d}", "C2", fact, date, url))

    c3 = [
        ("Apple reported fiscal 2018 revenue of $265.6 billion.", "2018-11-01", "https://www.apple.com/newsroom/2018/11/apple-reports-fourth-quarter-results/"),
        ("Microsoft reported fiscal 2019 revenue of $125.8 billion.", "2019-08-01", "https://www.microsoft.com/investor/reports/ar19/index.html"),
        ("The U.S. unemployment rate was 3.5 percent in September 2019.", "2019-10-04", "https://www.bls.gov/news.release/archives/empsit_10042019.htm"),
        ("Amazon reported 2018 net sales of $232.9 billion.", "2019-01-31", "https://ir.aboutamazon.com/news-release/news-release-details/2019/Amazoncom-Announces-Fourth-Quarter-Sales-up-20-to-724-Billion/default.aspx"),
        ("Alphabet reported 2019 revenue of $161.9 billion.", "2020-02-03", "https://abc.xyz/investor/news/news-details/2020/Alphabet-Announces-Fourth-Quarter-and-Fiscal-Year-2019-Results/"),
        ("Apple reported fiscal 2020 revenue of $274.5 billion.", "2020-10-29", "https://www.apple.com/newsroom/2020/10/apple-reports-fourth-quarter-results/"),
        ("The U.S. unemployment rate was 14.7 percent in April 2020.", "2020-05-08", "https://www.bls.gov/news.release/archives/empsit_05082020.htm"),
        ("Microsoft reported fiscal 2021 revenue of $168.1 billion.", "2021-07-27", "https://www.microsoft.com/en-us/Investor/earnings/FY-2021-Q4/press-release-webcast"),
        ("Meta reported 2022 revenue of $116.6 billion.", "2023-02-01", "https://investor.atmeta.com/investor-news/press-release-details/2023/Meta-Reports-Fourth-Quarter-and-Full-Year-2022-Results/default.aspx"),
        ("Apple reported fiscal 2023 revenue of $383.3 billion.", "2023-11-02", "https://www.apple.com/newsroom/2023/11/apple-reports-fourth-quarter-results/"),
    ]
    for idx, (fact, date, url) in enumerate(c3, 1):
        rows.append(item(f"c3_{idx:02d}", "C3", fact, date, url))

    c4 = [
        ("In 2018, Apple's revenue exceeded Microsoft's revenue.", "2019-08-01", "https://www.microsoft.com/investor/reports/ar19/index.html"),
        ("In September 2019, the U.S. unemployment rate was lower than in September 2018.", "2019-10-04", "https://www.bls.gov/news.release/archives/empsit_10042019.htm"),
        ("Apple's fiscal 2018 revenue exceeded Amazon's 2018 net sales.", "2019-01-31", "https://ir.aboutamazon.com/news-release/news-release-details/2019/Amazoncom-Announces-Fourth-Quarter-Sales-up-20-to-724-Billion/default.aspx"),
        ("Microsoft's fiscal 2019 revenue was lower than Alphabet's 2019 revenue.", "2020-02-03", "https://abc.xyz/investor/news/news-details/2020/Alphabet-Announces-Fourth-Quarter-and-Fiscal-Year-2019-Results/"),
        ("The April 2020 U.S. unemployment rate was higher than the September 2019 rate.", "2020-05-08", "https://www.bls.gov/news.release/archives/empsit_05082020.htm"),
        ("Apple's fiscal 2020 revenue exceeded its fiscal 2018 revenue.", "2020-10-29", "https://www.apple.com/newsroom/2020/10/apple-reports-fourth-quarter-results/"),
        ("Microsoft's fiscal 2021 revenue exceeded Alphabet's 2019 revenue.", "2021-07-27", "https://www.microsoft.com/en-us/Investor/earnings/FY-2021-Q4/press-release-webcast"),
        ("Meta's 2022 revenue was lower than Microsoft's fiscal 2021 revenue.", "2023-02-01", "https://investor.atmeta.com/investor-news/press-release-details/2023/Meta-Reports-Fourth-Quarter-and-Full-Year-2022-Results/default.aspx"),
        ("Apple's fiscal 2023 revenue exceeded its fiscal 2020 revenue.", "2023-11-02", "https://www.apple.com/newsroom/2023/11/apple-reports-fourth-quarter-results/"),
        ("The September 2019 unemployment rate was lower than the April 2020 rate.", "2020-05-08", "https://www.bls.gov/news.release/archives/empsit_05082020.htm"),
    ]
    for idx, (fact, date, url) in enumerate(c4, 1):
        rows.append(item(f"c4_{idx:02d}", "C4", fact, date, url))

    c5 = [
        ("The United Kingdom voted to leave the European Union on June 23, 2016.", "2016-06-23", "https://en.wikipedia.org/wiki/2016_United_Kingdom_European_Union_membership_referendum"),
        ("The United States inaugurated Donald Trump as president on January 20, 2017.", "2017-01-20", "https://en.wikipedia.org/wiki/Inauguration_of_Donald_Trump"),
        ("Apple released the iPhone X on November 3, 2017.", "2017-11-03", "https://www.apple.com/newsroom/2017/09/the-future-is-here-iphone-x/"),
        ("Microsoft won the JEDI cloud contract on October 25, 2019.", "2019-10-25", "https://www.defense.gov/News/Releases/Release/Article/1999635/department-of-defense-awards-jedi-cloud-contract/"),
        ("The World Health Organization was informed of pneumonia cases in Wuhan on December 31, 2019.", "2019-12-31", "https://www.who.int/emergencies/diseases/novel-coronavirus-2019/interactive-timeline"),
        ("The World Health Organization declared COVID-19 a pandemic on March 11, 2020.", "2020-03-11", "https://www.who.int/director-general/speeches/detail/who-director-general-s-opening-remarks-at-the-media-briefing-on-covid-19---11-march-2020"),
        ("The United Kingdom left the European Union on January 31, 2020.", "2020-01-31", "https://en.wikipedia.org/wiki/Brexit"),
        ("Joe Biden was inaugurated as U.S. president on January 20, 2021.", "2021-01-20", "https://en.wikipedia.org/wiki/Inauguration_of_Joe_Biden"),
        ("Russia launched a full-scale invasion of Ukraine on February 24, 2022.", "2022-02-24", "https://en.wikipedia.org/wiki/Russian_invasion_of_Ukraine"),
        ("Elon Musk completed the acquisition of Twitter on October 27, 2022.", "2022-10-27", "https://en.wikipedia.org/wiki/Acquisition_of_Twitter_by_Elon_Musk"),
    ]
    for idx, (fact, date, url) in enumerate(c5, 1):
        rows.append(item(f"c5_{idx:02d}", "C5", fact, date, url))

    c6 = [
        ("Apple's November 1, 2018 earnings release reported fiscal 2018 revenue of $265.6 billion.", "2018-11-01", "https://www.apple.com/newsroom/2018/11/apple-reports-fourth-quarter-results/"),
        ("The October 4, 2019 BLS employment report stated that unemployment was 3.5 percent.", "2019-10-04", "https://www.bls.gov/news.release/archives/empsit_10042019.htm"),
        ("Microsoft's August 1, 2019 annual report stated fiscal 2019 revenue of $125.8 billion.", "2019-08-01", "https://www.microsoft.com/investor/reports/ar19/index.html"),
        ("The WHO's December 31, 2019 timeline entry recorded pneumonia cases in Wuhan.", "2019-12-31", "https://www.who.int/emergencies/diseases/novel-coronavirus-2019/interactive-timeline"),
        ("Alphabet's February 3, 2020 release reported 2019 revenue of $161.9 billion.", "2020-02-03", "https://abc.xyz/investor/news/news-details/2020/Alphabet-Announces-Fourth-Quarter-and-Fiscal-Year-2019-Results/"),
        ("The WHO's March 11, 2020 briefing characterized COVID-19 as a pandemic.", "2020-03-11", "https://www.who.int/director-general/speeches/detail/who-director-general-s-opening-remarks-at-the-media-briefing-on-covid-19---11-march-2020"),
        ("Apple's October 29, 2020 earnings release reported fiscal 2020 revenue of $274.5 billion.", "2020-10-29", "https://www.apple.com/newsroom/2020/10/apple-reports-fourth-quarter-results/"),
        ("Microsoft's July 27, 2021 release reported fiscal 2021 revenue of $168.1 billion.", "2021-07-27", "https://www.microsoft.com/en-us/Investor/earnings/FY-2021-Q4/press-release-webcast"),
        ("Meta's February 1, 2023 release reported 2022 revenue of $116.6 billion.", "2023-02-01", "https://investor.atmeta.com/investor-news/press-release-details/2023/Meta-Reports-Fourth-Quarter-and-Full-Year-2022-Results/default.aspx"),
        ("Apple's November 2, 2023 earnings release reported fiscal 2023 revenue of $383.3 billion.", "2023-11-02", "https://www.apple.com/newsroom/2023/11/apple-reports-fourth-quarter-results/"),
    ]
    for idx, (fact, date, url) in enumerate(c6, 1):
        rows.append(item(f"c6_{idx:02d}", "C6", fact, date, url))

    outcomes = [
        ("The United Kingdom voted to leave the European Union.", "whether the United Kingdom would vote to leave the European Union", "2016-06-23"),
        ("Donald Trump won the 2016 U.S. presidential election.", "whether Donald Trump would win the 2016 U.S. presidential election", "2016-11-08"),
        ("Microsoft won the JEDI cloud contract.", "which company would win the JEDI cloud contract", "2019-10-25"),
        ("The United Kingdom left the European Union.", "whether the United Kingdom would leave the European Union", "2020-01-31"),
        ("The WHO declared COVID-19 a pandemic.", "whether the WHO would declare COVID-19 a pandemic", "2020-03-11"),
        ("Joe Biden won the 2020 U.S. presidential election.", "whether Joe Biden would win the 2020 U.S. presidential election", "2020-11-07"),
        ("Joe Biden was inaugurated as U.S. president.", "whether Joe Biden would be inaugurated as U.S. president", "2021-01-20"),
        ("Russia launched a full-scale invasion of Ukraine.", "whether Russia would launch a full-scale invasion of Ukraine", "2022-02-24"),
        ("Elon Musk completed the acquisition of Twitter.", "whether Elon Musk would complete the acquisition of Twitter", "2022-10-27"),
        ("Argentina won the 2022 FIFA World Cup.", "which country would win the 2022 FIFA World Cup", "2022-12-18"),
    ]
    for idx, (fact, target, date) in enumerate(outcomes, 1):
        rows.append(
            item(
                f"c7_{idx:02d}",
                "C7",
                fact,
                date,
                "https://en.wikipedia.org/wiki/Main_Page",
                cutoff=f"{int(date[:4]) - 1:04d}-01-01",
                target=target,
            )
        )

    consequences = [
        ("The Brexit referendum result led David Cameron to announce his resignation.", "whether the United Kingdom would vote to leave the European Union", "2016-06-24"),
        ("Donald Trump's election victory initiated his presidential transition.", "whether Donald Trump would win the 2016 U.S. presidential election", "2016-11-09"),
        ("The JEDI award triggered a legal challenge by Amazon.", "which company would win the JEDI cloud contract", "2019-11-22"),
        ("The United Kingdom's departure began the post-Brexit transition period.", "whether the United Kingdom would leave the European Union", "2020-02-01"),
        ("The pandemic declaration led governments to intensify public-health restrictions.", "whether the WHO would declare COVID-19 a pandemic", "2020-03-12"),
        ("Joe Biden's election victory initiated his presidential transition.", "whether Joe Biden would win the 2020 U.S. presidential election", "2020-11-08"),
        ("Joe Biden's inauguration made Kamala Harris vice president.", "whether Joe Biden would be inaugurated as U.S. president", "2021-01-20"),
        ("Russia's invasion led to extensive international sanctions.", "whether Russia would launch a full-scale invasion of Ukraine", "2022-02-25"),
        ("Musk's Twitter acquisition made the company privately held.", "whether Elon Musk would complete the acquisition of Twitter", "2022-10-28"),
        ("Argentina's World Cup victory prompted championship celebrations in Buenos Aires.", "which country would win the 2022 FIFA World Cup", "2022-12-19"),
    ]
    for idx, (fact, target, date) in enumerate(consequences, 1):
        rows.append(
            item(
                f"c8_{idx:02d}",
                "C8",
                fact,
                date,
                "https://en.wikipedia.org/wiki/Main_Page",
                cutoff=f"{int(date[:4]) - 1:04d}-01-01",
                target=target,
            )
        )

    return rows


def build_boundary_items(taxonomy_rows: list[dict]) -> list[dict]:
    publications = [row for row in taxonomy_rows if row["gold_category"] == "C6"]
    rows: list[dict] = []
    for idx, base in enumerate(publications, 1):
        date = base["gold_effective_source_date"]
        assert date is not None
        year = int(date[:4])
        before = f"{year - 1:04d}-01-01"
        after = f"{year + 1:04d}-12-31"
        rows.append(
            item(
                f"boundary_{idx:02d}_before",
                "C6",
                base["fact"],
                date,
                base["source_url"],
                cutoff=before,
                subset="date_boundary",
            )
        )
        rows.append(
            item(
                f"boundary_{idx:02d}_after",
                "C6",
                base["fact"],
                date,
                base["source_url"],
                cutoff=after,
                subset="date_boundary",
            )
        )
    # Add ten more safe/leaked pairs by reusing five authoritative publications.
    for idx, base in enumerate(publications[:5], 11):
        date = base["gold_effective_source_date"]
        assert date is not None
        year = int(date[:4])
        rows.append(
            item(
                f"boundary_{idx:02d}_before",
                "C6",
                base["fact"],
                date,
                base["source_url"],
                cutoff=f"{year - 2:04d}-06-30",
                subset="date_boundary",
            )
        )
        rows.append(
            item(
                f"boundary_{idx:02d}_after",
                "C6",
                base["fact"],
                date,
                base["source_url"],
                cutoff=f"{year + 2:04d}-06-30",
                subset="date_boundary",
            )
        )
    return rows


def main() -> None:
    taxonomy = build_taxonomy_items()
    boundary = build_boundary_items(taxonomy)
    rows = taxonomy + boundary
    assert len(taxonomy) == 90
    assert len(boundary) == 30
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} claims to {OUT}")


if __name__ == "__main__":
    main()
