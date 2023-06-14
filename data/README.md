# Dataset Descriptions

All datasets are formatted to contain variable region heavy chain, variable region light chain, and one or more fitness metric.

## Folder breakup

1. `binding` - Binding affinity datasets from [Hie et al.](https://www.nature.com/articles/s41587-023-01763-2), and [Koenig et al.](https://www.pnas.org/doi/10.1073/pnas.1613231114?url_ver=Z39.88-2003&rfr_id=ori:rid:crossref.org&rfr_dat=cr_pub%20%200pubmed), and [Warszawski et al.](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007207), where fitness is reported in Kd (nM)
2. `expression` - Expression dataset from [Koenig et al.](https://www.pnas.org/doi/10.1073/pnas.1613231114?url_ver=Z39.88-2003&rfr_id=ori:rid:crossref.org&rfr_dat=cr_pub%20%200pubmed), where fitness is reported as the enrichment ratio (ER) relative to wildtype
3. `immunogenicity` - Immunogenicity datasets from [Prihoda et al.](https://www.tandfonline.com/doi/full/10.1080/19420862.2021.2020203), where fitness is reported as the percentage of patients that elicit an anti-drug antibody response when administered with an antibody therapeutic.
4. `thermostability` - Thermostability datasets from [Hie et al.](https://www.nature.com/articles/s41587-023-01763-2), where fitness is reported as the melting temperature (°C)
5. `wittrup` - Therapeutic antibody candidates from [Jain et al.](https://www.pnas.org/doi/10.1073/pnas.1616408114), where fitness is reported with 12 biophysical assays (HEK Titer, Tm, SGAC-SINS, HIC Retention Time, SMAC Retention Time, Slope for Accelerated Stability, PSR SMP, AC-SINS, CIC Retention Time, CSI-BLI, ELISA, and BVP ELISA)
6. `cst` - Clinical Stage Therapeutics used for setting the [Therapeutic Antibody Profiler](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabpred/tap) guidelines in [Raybould et al.](https://www.pnas.org/doi/10.1073/pnas.1810576116)
