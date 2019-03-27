import kfp
import kfp.compiler as compiler
import kfp.dsl as dsl
from kfp_experiment.models.api_experiment import ApiExperiment
from kubernetes import client as k8s_client, config
from kubernetes.config import ConfigException


def preprocess_op():
    return dsl.ContainerOp(
        name='preprocess',
        image='romibuzi/kubeflow-mnist:preprocessing-0.0.1',
        arguments='',
        file_outputs={
            'output': '/output.txt'
        }
    )


def train_op(preprocess_output: str, epoch: int, dropout: float, hidden_layer_size: int):
    return dsl.ContainerOp(
        name='train',
        image='romibuzi/kubeflow-mnist:train-0.0.1',
        arguments=[
            '--preprocess-output', preprocess_output,
            '--epoch', str(epoch),
            '--dropout', str(dropout),
            '--hidden-layer-size', str(hidden_layer_size)
        ],
        file_outputs={'output': '/output.txt'}
    )


def prediction_op(train_output: str, preprocess_output: str, cm_bucket_name: str, cm_path: str):
    return dsl.ContainerOp(
        name='prediction',
        image='romibuzi/kubeflow-mnist:prediction-0.0.1',
        arguments=[
            '--preprocess-output', preprocess_output,
            '--train-output', train_output,
            '--bucket-name', cm_bucket_name,
            '--cm-path', cm_path
        ]
    )


def kubeflow_deploy_op(train_output: str, tf_server_name: str, step_name='deploy'):
    return dsl.ContainerOp(
        name=step_name,
        image='romibuzi/kubeflow-mnist:deploy-0.0.2',
        arguments=[
            '--cluster-name', 'mnist-pipeline',
            '--namespace', 'kubeflow',
            '--train-output', train_output,
            '--server-name', tf_server_name,
            '--pvc-name', 'workflow-pvc'
        ]
    )


def find_minio_ip():
    try:
        config.load_incluster_config()
    except ConfigException:
        config.load_kube_config()

    v1 = k8s_client.CoreV1Api()
    response = v1.list_service_for_all_namespaces(watch=False)

    for service in response.items:
        if service.metadata.name == "minio-service":
            return service.spec.cluster_ip
    raise Exception("minio service not found !")


@dsl.pipeline(
    name='pipeline pipeline-mnist',
    description=''
)
def mnist(cm_bucket_name: str,
          cm_path: str,
          epoch: int = 5,
          dropout: float = 0.2,
          hidden_layer_size: int = 512,
          deploy: bool = False):
    pvc = k8s_client.V1PersistentVolumeClaimVolumeSource(claim_name='workflow-pvc')
    volume = k8s_client.V1Volume(name='workflow-nfs',
                                 persistent_volume_claim=pvc)
    volume_mount = k8s_client.V1VolumeMount(mount_path='/mnt', name='workflow-nfs')

    s3_endpoint = k8s_client.V1EnvVar('S3_ENDPOINT', f'http://{find_minio_ip()}:9000')
    s3_access_key = k8s_client.V1EnvVar('AWS_ACCESS_KEY_ID', 'minio')
    s3_secret_key = k8s_client.V1EnvVar('AWS_SECRET_ACCESS_KEY', 'minio123')

    preprocess = preprocess_op() \
        .add_volume(volume)\
        .add_volume_mount(volume_mount)

    train = train_op(preprocess.output, epoch, dropout, hidden_layer_size) \
        .add_volume_mount(volume_mount)

    predictions = prediction_op(train.output, preprocess.output, cm_bucket_name, cm_path) \
        .add_volume_mount(volume_mount) \
        .add_env_variable(s3_endpoint) \
        .add_env_variable(s3_access_key) \
        .add_env_variable(s3_secret_key)

    if deploy:
        kubeflow_deploy_op(train.output, 'mnist-pipeline-{{workflow.name}}') \
           .add_volume_mount(volume_mount)


def get_or_create_experiment(experiment_name: str, client: kfp.Client) -> ApiExperiment:
    existing_experiments = client.list_experiments().experiments

    if existing_experiments is not None:
        exp = next(iter([exp for exp in existing_experiments if exp.name == experiment_name]), None)
    else:
        exp = None

    if exp is None:
        exp = client.create_experiment(experiment_name)
        print('Experiment %s created with ID %s' % (exp.name, exp.id))
    else:
        print('Experiment already exists with id %s' % exp.id)

    return exp


def main():
    experiment_name = 'mnist-pipeline'
    pipeline_func = mnist
    pipeline_filename = pipeline_func.__name__ + '.tar.gz'

    compiler.Compiler().compile(pipeline_func, pipeline_filename)

    client = kfp.Client()
    experiment = get_or_create_experiment(experiment_name, client)

    # Submit a pipeline run
    pipeline_arguments = {
        'cm_bucket_name': 'sleq-ml',
        'cm_path': 'metrics/cm.csv.tar.gz',
        'epoch': 5,
        'dropout': 0.2,
        'hidden_layer_size': 512,
        'deploy': True
    }
    run_name = pipeline_func.__name__ + ' run'
    run_result = client.run_pipeline(experiment.id, run_name, pipeline_filename, pipeline_arguments)


main()
