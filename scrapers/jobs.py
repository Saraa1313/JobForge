import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import random

MONTHS_TO_SCRAPE = 8 



def get_recent_hiring_threads(limit=8):
    """
    Dynamically fetches recent 'Who is hiring?' thread IDs via Algolia API.
    """

    url = "http://hn.algolia.com/api/v1/search_by_date"
    params = {
        "tags": "story,author_whoishiring",
        "query": "Who is hiring?",
        "hitsPerPage": limit
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()
        threads = []

        for hit in data["hits"]:
            if "Who is hiring?" in hit["title"]:
                threads.append({
                    "id": hit["objectID"],
                    "title": hit["title"]
                })

        return threads

    except Exception as e:
        print(f"Error fetching hiring threads: {e}")
        return []



def classify_job_role(text):
    """
    Labels the job posting into a bucket: SWE / ML / Hardware / DevOps / Data Eng
    """
    t = text.lower()

    hardware_keywords = [
        'hardware', 'firmware', 'embedded', 'fpga', 'verilog',
        'vhdl', 'asic', 'pcb', 'electrical engineer', 'robotics',
        'iot', 'driver'
    ]
    if any(k in t for k in hardware_keywords):
        return 'Hardware / Embedded'

    ml_keywords = [
        'machine learning', 'computer vision', 'nlp', 'pytorch',
        'tensorflow', 'data scientist', 'ai engineer', 'deep learning',
        'llm', 'generative ai'
    ]
    if any(k in t for k in ml_keywords):
        return 'ML & Data Science'

    devops_keywords = [
        'devops', 'sre', 'site reliability', 'kubernetes', 'terraform',
        'infrastructure', 'ci/cd', 'aws', 'cloud engineer', 'platform engineer'
    ]
    if any(k in t for k in devops_keywords):
        return 'DevOps & Cloud'

    data_eng_keywords = [
        'data engineer', 'etl', 'pipeline', 'spark',
        'hadoop', 'kafka', 'warehouse', 'snowflake'
    ]
    if any(k in t for k in data_eng_keywords):
        return 'Data Engineering'

    return 'Software Engineer'



def scrape_thread(thread_id):

    jobs = []
    page = 1

    while True:
        url = f"https://news.ycombinator.com/item?id={thread_id}&p={page}"

        try:
            time.sleep(random.uniform(0.5, 1.3))
            resp = requests.get(url, timeout=10)

            if resp.status_code != 200:
                break

            soup = BeautifulSoup(resp.text, "html.parser")
            comments = soup.find_all("tr", class_="athing comtr")

            if not comments:
                break

            found_on_page = 0

            for comment in comments:

                indent = comment.find("td", class_="ind")
                if indent and indent.find("img", attrs={"width": "0"}):

                    body = comment.find("div", class_="commtext")
                    if not body:
                        continue

                    text = body.get_text(separator=" ", strip=True)

                    if len(text) < 40:
                        continue

                    job_id = comment["id"]
                    role = classify_job_role(text)
                    job_link = f"https://news.ycombinator.com/item?id={job_id}"

                    jobs.append({
                        "job_id": job_id,
                        "job_role": role,
                        "job_desc": text,
                        "job_link": job_link
                    })

                    found_on_page += 1


            if not soup.find("a", string="More"):
                break

            page += 1

        except Exception as e:
            print(f"   Error on page {page}: {e}")
            break

    return jobs


if __name__ == "__main__":

    threads = get_recent_hiring_threads(limit=8)
    all_jobs = []

    for t in threads:
        scraped = scrape_thread(t["id"])
        all_jobs.extend(scraped)

    df = pd.DataFrame(all_jobs)

    if df.empty:
        print("\n✗ No jobs scraped.")
        exit()

    df["fingerprint"] = df["job_desc"].str[:150]
    df = df.drop_duplicates(subset="fingerprint", keep="last")
    df = df.drop(columns=["fingerprint"])

 
    out = "hn_job_dataset_latest.csv"
    df.to_csv(out, index=False)

    print(f"\n Saved dataset")

