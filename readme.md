# Kubeflow

## Install Kubernetes cluster

[Instruction to setup account](https://www.kubeflow.org/docs/started/getting-started-gke/)

- [Configurer l'écran d'autorisation](https://console.cloud.google.com/apis/credentials/consent?project=kubeflow-227910&folder&organizationId=552523943544)
- Ajouter Credential
  - ClientID : 559752670223-4des59ecd2pe72b49p99tdhgmvs6etfl.apps.googleusercontent.com
  - Secret : NQ_Ce7P9TM8dQ8PTVSfXRQrW


## Install Kubeflow

Using the web ui:  https://deploy.kubeflow.cloud/#/deploy => OK

OU

```bash
export CLIENT_ID=559752670223-4des59ecd2pe72b49p99tdhgmvs6etfl.apps.googleusercontent.com
export CLIENT_SECRET=NQ_Ce7P9TM8dQ8PTVSfXRQrW
export PROJECT=kubeflow-227910
export DEPLOYMENT_NAME=kubeflow-app
export KUBEFLOW_VERSION=0.4.1
curl https://raw.githubusercontent.com/kubeflow/kubeflow/v${KUBEFLOW_VERSION}/scripts/gke/deploy.sh | bash


kubectl -n kubeflow get all
```

https://kubeflow-app.endpoints.kubeflow-227910.cloud.goog/


## Import Data

Import des données à la main dans un bucket GCS



## Explore Jupyter

Dans un notebook :
- `!gsutil -m rsync -r gs://kubeflow-sleq ./work/`


## Run simple tensorflow script






## Run distributed tensorflow script






## Deploy model





## Upgrade model






