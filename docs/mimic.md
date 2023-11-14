
## [MIMIC III](<https://physionet.org/content/mimici> ii/1.4/)

### Scripts

```bash
cd data
# download MIMIC III v1.4
wget -r -N -c -np --user gusui --ask-password https://physionet.org/files/mimiciii/1.4/
```

### References

- <https://github.com/mlds-lab/interp-net/blob/master/src/mimic_data_extraction.py>

## [MIMIC IV](https://physionet.org/content/mimiciv/2.2/)

### Scripts

```bash
cd data
# download MIMIC IV v2.2, 7.2G
wget -r -N -c -np --user gusui --ask-password https://physionet.org/files/mimiciv/2.2/

# install postgresql
# https://www.postgresql.org/download/linux/ubuntu/
sudo apt-get update
sudo apt-get -y install postgresql

# load mimic into postgresql
# https://github.com/MIT-LCP/mimic-code/tree/v2.4.0/mimic-iv/buildmimic/postgres
git clone https://github.com/MIT-LCP/mimic-code.git
cd mimic-code/
mv ../physionet.org/files/mimiciv mimiciv

# To confirm
sudo -u postgres createuser -s $(whoami)
createdb mimiciv
psql -d mimiciv -f mimic-iv/buildmimic/postgres/create.sql
psql -d mimiciv -v ON_ERROR_STOP=1 -v mimic_data_dir=mimiciv/2.2 -f mimic-iv/buildmimic/postgres/load_gz.sql
psql -d mimiciv -v ON_ERROR_STOP=1 -v mimic_data_dir=mimiciv/2.2 -f mimic-iv/buildmimic/postgres/constraint.sql
psql -d mimiciv -v ON_ERROR_STOP=1 -v mimic_data_dir=mimiciv/2.2 -f mimic-iv/buildmimic/postgres/index.sql


# https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/concepts#generating-the-concepts-on-postgresql


# dropdb mimiciv

# Genereate concepts
psql -d mimiciv -f mimic-iv/concepts_postgres/treatment/ventilation.sql
psql -d mimiciv -f mimic-iv/concepts_postgres/medication/vasopressin.sql

cd mimic-iv/concepts_postgres/
psql -d mimiciv -f postgres-functions.sql
psql -d mimiciv -f postgres-make-concepts.sql
```
