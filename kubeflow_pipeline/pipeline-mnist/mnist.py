import kfp
import kfp.compiler as compiler
import kfp.dsl as dsl
from kfp_experiment.models.api_experiment import ApiExperiment
from kubernetes import client as k8s_client


def preprocess_op():
    return dsl.ContainerOp(
        name='preprocess',
        image='romibuzi/kubeflow-mnist:preprocessing-fifth',
        arguments=[],
        file_outputs={
            'output': '/output.txt'
        }
    )


def train_op(preprocess_output: str):
    return dsl.ContainerOp(
        name='train',
        image='romibuzi/kubeflow-mnist:train-seventh',
        arguments=[
            '--preprocess-output', preprocess_output
        ],
        file_outputs={'output': '/output.txt'}
    )


def prediction_op(train_output: str, preprocess_output: str, cm_bucket_name: str, cm_path: str):
    return dsl.ContainerOp(
        name='prediction',
        image='romibuzi/kubeflow-mnist:prediction-third',
        arguments=[
            '--preprocess-output', preprocess_output,
            '--train-output', train_output,
            '--bucket-name', cm_bucket_name,
            '--cm-path', cm_path
        ],
        file_outputs={'prediction': '/output.txt'}
    )


@dsl.pipeline(
    name='pipeline pipeline-mnist',
    description=''
)
def mnist(cm_bucket_name: str, cm_path: str):
    pvc = k8s_client.V1PersistentVolumeClaimVolumeSource(claim_name='workflow-pvc')
    volume = k8s_client.V1Volume(name='workflow-nfs',
                                 persistent_volume_claim=pvc)
    volume_mount = k8s_client.V1VolumeMount(mount_path='/mnt', name='workflow-nfs')
    s3_endpoint = k8s_client.V1EnvVar('S3_ENDPOINT', 'http://10.101.144.113:9000')
    s3_access_key = k8s_client.V1EnvVar('AWS_ACCESS_KEY_ID', 'minio')
    s3_secret_key = k8s_client.V1EnvVar('AWS_SECRET_ACCESS_KEY', 'minio123')

    preprocess = preprocess_op() \
        .add_volume(volume)\
        .add_volume_mount(volume_mount)

    train = train_op(preprocess.output) \
        .add_volume_mount(volume_mount)

    predictions = prediction_op(train.output, preprocess.output, cm_bucket_name, cm_path) \
        .add_volume_mount(volume_mount) \
        .add_env_variable(s3_endpoint) \
        .add_env_variable(s3_access_key) \
        .add_env_variable(s3_secret_key)


def get_or_create_experiment(experiment_name: str) -> ApiExperiment:
    existing_experiments = client.list_experiments().experiments

    if existing_experiments is not None:
        experiment = next(iter([exp for exp in existing_experiments if exp.name == EXPERIMENT_NAME]), None)
    else:
        experiment = None

    if experiment is None:
        experiment = client.create_experiment(experiment_name)
        print('Experiment %s created with ID %s' % (experiment.name, experiment.id))
    else:
        print('Experiment already exists with id %s' % experiment.id)

    return experiment


EXPERIMENT_NAME = 'pipeline-mnist with train'
pipeline_func = mnist
pipeline_filename = pipeline_func.__name__ + '.tar.gz'

compiler.Compiler().compile(pipeline_func, pipeline_filename)

client = kfp.Client()
experiment = get_or_create_experiment(EXPERIMENT_NAME)

# Submit a pipeline run
pipeline_arguments = { 'cm_bucket_name': 'sleq-ml', 'cm_path': 'metrics/cm.csv.tar.gz'}
run_name = pipeline_func.__name__ + ' run'
run_result = client.run_pipeline(experiment.id, run_name, pipeline_filename, pipeline_arguments)
