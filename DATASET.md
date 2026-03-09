# Dataset provenance

The supplied file is `WA_Fn-UseC_-Telco-Customer-Churn.csv`.
It has 7,043 customer rows, 19 model inputs, a customer identifier,
and the binary `Churn` outcome. The source file is preserved unchanged.

This schema matches the widely distributed IBM Telco Customer Churn
sample, also available through Kaggle. The file was supplied by the user;
this project does not claim it was obtained directly from Kaggle's API.

- [IBM sample project](https://github.com/IBM/telco-customer-churn-on-icp4d)
- [Kaggle dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- [IBM description of the fictional Telco sample family](https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113)

The IBM enhanced sample family contains additional narrative columns.
This project uses only the 21-column CSV provided here, not those enhanced
fields. There is no timestamp, actual retention offer, treatment assignment,
or observed intervention outcome in this CSV. It is an educational sample,
not evidence of a live telecom customer feed.

The audit records the original file's SHA-256 in `reports/run_summary.json`.
`TotalCharges` contains 11 blank strings, converted to numeric missing
values. These rows remain in the experiment. A training-only median and
missingness indicator handle them. `customerID` is never a model feature.

For public redistribution, preserve source attribution and review the
upstream dataset terms. No new ownership claim is made over the dataset.
