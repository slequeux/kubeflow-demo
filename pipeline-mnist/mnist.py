import kfp
import kfp.compiler as compiler
import kfp.dsl as dsl
from kfp_experiment.models.api_experiment import ApiExperiment
from kubernetes import client as k8s_client


def preprocess_op():
    return dsl.ContainerOp(
        name='preprocess',
        image='romibuzi/kubeflow-pipeline-mnist:preprocessing-fifth',
        arguments=[],
        file_outputs={
            'output': '/output.txt'
        }
    )


def train_op(preprocess_output: str):
    return dsl.ContainerOp(
        name='train',
        image='romibuzi/kubeflow-pipeline-mnist:train-seventh',
        arguments=[
            '--preprocess-output', preprocess_output
        ],
        file_outputs={'output': '/output.txt'}
    )


def prediction_op(train_output: str, preprocess_output: str):
    return dsl.ContainerOp(
        name='prediction',
        image='romibuzi/kubeflow-pipeline-mnist:prediction-third',
        arguments=[
            '--preprocess-output', preprocess_output,
            '--train-output', train_output
        ],
        file_outputs={'prediction': '/output.txt'}
    )


@dsl.pipeline(
    name='pipeline pipeline-mnist',
    description=''
)
def mnist():
    preprocess = preprocess_op().add_volume(
        k8s_client.V1Volume(name='workflow-nfs',
                            persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(claim_name='workflow-pvc'))
    ).add_volume_mount(k8s_client.V1VolumeMount(mount_path='/mnt', name='workflow-nfs'))

    train = train_op(preprocess.output).add_volume_mount(
        k8s_client.V1VolumeMount(mount_path='/mnt', name='workflow-nfs')
    )

    predictions = prediction_op(train.output, preprocess.output).add_volume_mount(
        k8s_client.V1VolumeMount(mount_path='/mnt', name='workflow-nfs')
    )


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
run_name = pipeline_func.__name__ + ' run'
run_result = client.run_pipeline(experiment.id, run_name, pipeline_filename)
