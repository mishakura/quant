import requests

CIK = "0001318605"
HEADERS = {"User-Agent": "Your Name your@email.com"}

def fetch_sec_data(concept):
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{CIK.zfill(10)}/us-gaap/{concept}.json"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return resp.json()

def get_quarter_end_dates(year):
    return [
        f"{year}-03-31",
        f"{year}-06-30",
        f"{year}-09-30",
        f"{year}-12-31"
    ]

def organize(data, unit):
    d = {}
    for item in data['units'][unit]:
        end = item['end']
        fp = item.get('fp', '')
        d[(end, fp)] = item['val']
    return d

def main():
    eps_data = fetch_sec_data("EarningsPerShareBasic")
    net_income_data = fetch_sec_data("NetIncomeLoss")
    shares_data = fetch_sec_data("WeightedAverageNumberOfSharesOutstandingBasic")

    eps_dict = organize(eps_data, 'USD/shares')
    ni_dict = organize(net_income_data, 'USD')
    sh_dict = organize(shares_data, 'shares')

    years = set()
    for (end, fp) in list(eps_dict.keys()) + list(ni_dict.keys()) + list(sh_dict.keys()):
        years.add(end[:4])

    quarter_eps = {}

    for year in sorted(years):
        q_ends = get_quarter_end_dates(year)

        # Q1
        if (q_ends[0], 'Q1') in eps_dict:
            quarter_eps[q_ends[0]] = {'eps': eps_dict[(q_ends[0], 'Q1')], 'source': 'actual Q1 EPS'}
        elif (q_ends[0], 'Q1') in ni_dict and (q_ends[0], 'Q1') in sh_dict:
            quarter_eps[q_ends[0]] = {'eps': ni_dict[(q_ends[0], 'Q1')] / sh_dict[(q_ends[0], 'Q1')], 'source': 'Q1 NI/Shares'}
        elif (q_ends[1], 'H1') in ni_dict and (q_ends[1], 'H1') in sh_dict:
            quarter_eps[q_ends[0]] = {'eps': (ni_dict[(q_ends[1], 'H1')] / 2) / sh_dict[(q_ends[1], 'H1')], 'source': 'H1 NI/Shares split'}
        elif (q_ends[1], 'H1') in eps_dict:
            quarter_eps[q_ends[0]] = {'eps': eps_dict[(q_ends[1], 'H1')] / 2, 'source': 'H1 EPS split'}

        # Q2
        if (q_ends[1], 'Q2') in eps_dict:
            quarter_eps[q_ends[1]] = {'eps': eps_dict[(q_ends[1], 'Q2')], 'source': 'actual Q2 EPS'}
        elif (q_ends[1], 'Q2') in ni_dict and (q_ends[1], 'Q2') in sh_dict:
            quarter_eps[q_ends[1]] = {'eps': ni_dict[(q_ends[1], 'Q2')] / sh_dict[(q_ends[1], 'Q2')], 'source': 'Q2 NI/Shares'}
        elif (q_ends[1], 'H1') in ni_dict and (q_ends[1], 'H1') in sh_dict and (q_ends[0], 'Q1') in ni_dict and (q_ends[0], 'Q1') in sh_dict:
            h1_ni = ni_dict[(q_ends[1], 'H1')]
            h1_sh = sh_dict[(q_ends[1], 'H1')]
            q1_ni = ni_dict[(q_ends[0], 'Q1')]
            q2_ni = h1_ni - q1_ni
            q2_eps = q2_ni / h1_sh
            quarter_eps[q_ends[1]] = {'eps': q2_eps, 'source': 'Q2 from H1-Q1 NI/Shares'}
        elif (q_ends[1], 'H1') in eps_dict and q_ends[0] in quarter_eps:
            h1_eps = eps_dict[(q_ends[1], 'H1')]
            q2_eps = h1_eps - quarter_eps[q_ends[0]]['eps']
            quarter_eps[q_ends[1]] = {'eps': q2_eps, 'source': 'Q2 from H1-Q1 EPS'}
        elif (q_ends[1], 'H1') in ni_dict and (q_ends[1], 'H1') in sh_dict:
            quarter_eps[q_ends[1]] = {'eps': (ni_dict[(q_ends[1], 'H1')] / 2) / sh_dict[(q_ends[1], 'H1')], 'source': 'H1 NI/Shares split'}
        elif (q_ends[1], 'H1') in eps_dict:
            quarter_eps[q_ends[1]] = {'eps': eps_dict[(q_ends[1], 'H1')] / 2, 'source': 'H1 EPS split'}

        # Q3
        if (q_ends[2], 'Q3') in eps_dict:
            quarter_eps[q_ends[2]] = {'eps': eps_dict[(q_ends[2], 'Q3')], 'source': 'actual Q3 EPS'}
        elif (q_ends[2], 'Q3') in ni_dict and (q_ends[2], 'Q3') in sh_dict:
            quarter_eps[q_ends[2]] = {'eps': ni_dict[(q_ends[2], 'Q3')] / sh_dict[(q_ends[2], 'Q3')], 'source': 'Q3 NI/Shares'}
        elif (q_ends[3], 'H2') in ni_dict and (q_ends[3], 'H2') in sh_dict and (q_ends[2], 'Q4') not in ni_dict:
            quarter_eps[q_ends[2]] = {'eps': (ni_dict[(q_ends[3], 'H2')] / 2) / sh_dict[(q_ends[3], 'H2')], 'source': 'H2 NI/Shares split'}
        elif (q_ends[3], 'H2') in eps_dict:
            quarter_eps[q_ends[2]] = {'eps': eps_dict[(q_ends[3], 'H2')] / 2, 'source': 'H2 EPS split'}

        # Q4
        if (q_ends[3], 'Q4') in eps_dict:
            quarter_eps[q_ends[3]] = {'eps': eps_dict[(q_ends[3], 'Q4')], 'source': 'actual Q4 EPS'}
        elif (q_ends[3], 'Q4') in ni_dict and (q_ends[3], 'Q4') in sh_dict:
            quarter_eps[q_ends[3]] = {'eps': ni_dict[(q_ends[3], 'Q4')] / sh_dict[(q_ends[3], 'Q4')], 'source': 'Q4 NI/Shares'}
        elif (q_ends[3], 'H2') in ni_dict and (q_ends[3], 'H2') in sh_dict and (q_ends[2], 'Q3') in ni_dict and (q_ends[2], 'Q3') in sh_dict:
            h2_ni = ni_dict[(q_ends[3], 'H2')]
            h2_sh = sh_dict[(q_ends[3], 'H2')]
            q3_ni = ni_dict[(q_ends[2], 'Q3')]
            q4_ni = h2_ni - q3_ni
            q4_eps = q4_ni / h2_sh
            quarter_eps[q_ends[3]] = {'eps': q4_eps, 'source': 'Q4 from H2-Q3 NI/Shares'}
        elif (q_ends[3], 'H2') in eps_dict and q_ends[2] in quarter_eps:
            h2_eps = eps_dict[(q_ends[3], 'H2')]
            q4_eps = h2_eps - quarter_eps[q_ends[2]]['eps']
            quarter_eps[q_ends[3]] = {'eps': q4_eps, 'source': 'Q4 from H2-Q3 EPS'}
        elif (q_ends[3], 'FY') in ni_dict and (q_ends[3], 'FY') in sh_dict and all(q in quarter_eps for q in q_ends[:3]):
            fy_ni = ni_dict[(q_ends[3], 'FY')]
            fy_sh = sh_dict[(q_ends[3], 'FY')]
            q1_ni = ni_dict.get((q_ends[0], 'Q1'))
            q2_ni = ni_dict.get((q_ends[1], 'Q2'))
            q3_ni = ni_dict.get((q_ends[2], 'Q3'))
            if q1_ni and q2_ni and q3_ni:
                q4_ni = fy_ni - (q1_ni + q2_ni + q3_ni)
                q4_eps = q4_ni / fy_sh
                quarter_eps[q_ends[3]] = {'eps': q4_eps, 'source': 'Q4 from FY-Q1-Q2-Q3 NI/Shares'}
            else:
                q4_eps = (fy_ni / 4) / fy_sh
                quarter_eps[q_ends[3]] = {'eps': q4_eps, 'source': 'Q4 from FY NI/Shares split'}
        elif (q_ends[3], 'FY') in eps_dict and all(q in quarter_eps for q in q_ends[:3]):
            fy_eps = eps_dict[(q_ends[3], 'FY')]
            q4_eps = fy_eps - sum(quarter_eps[q]['eps'] for q in q_ends[:3])
            quarter_eps[q_ends[3]] = {'eps': q4_eps, 'source': 'Q4 from FY-Q1-Q2-Q3 EPS'}
        elif (q_ends[3], 'FY') in eps_dict:
            q4_eps = eps_dict[(q_ends[3], 'FY')] / 4
            quarter_eps[q_ends[3]] = {'eps': q4_eps, 'source': 'Q4 from FY EPS split'}

    # Print all raw EPS data for debugging
    print("\n--- RAW EPS DATA ---")
    for item in eps_data['units']['USD/shares']:
        period = item.get('fp', 'N/A')
        end_date = item['end']
        value = item['val']
        print(f"End: {end_date} | Period: {period:>3} | Value: {value}")

    # Print all non-quarterly periods for debugging
    print("\n--- NON-QUARTERLY PERIODS IN EPS DATA ---")
    for item in eps_data['units']['USD/shares']:
        period = item.get('fp', 'N/A')
        end_date = item['end']
        value = item['val']
        if period not in ('Q1', 'Q2', 'Q3', 'Q4'):
            print(f"End: {end_date} | Period: {period:>3} | Value: {value}")

    # Print final quarterly EPS (one per quarter end date)
    print("\n--- FINAL QUARTERLY EPS (ONE PER QUARTER) ---")
    for end in sorted(quarter_eps.keys()):
        entry = quarter_eps[end]
        print(f"{end} | EPS: {entry['eps']:.4f} | {entry['source']}")

if __name__ == "__main__":
    main()