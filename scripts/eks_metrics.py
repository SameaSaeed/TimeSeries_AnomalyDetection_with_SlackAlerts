import boto3
import pandas as pd
from datetime import datetime, timedelta

cw = boto3.client('cloudwatch')

def get_eks_training_data(cluster_name, days=14):
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)
    
    # EKS Container Insights namespaces and metrics
    queries = [
        {
            'Id': 'cpu',
            'MetricStat': {
                'Metric': {
                    'Namespace': 'ContainerInsights',
                    'MetricName': 'node_cpu_utilization',
                    'Dimensions': [{'Name': 'ClusterName', 'Value': cluster_name}]
                },
                'Period': 3600,
                'Stat': 'Average'
            }
        },
        {
            'Id': 'disk',
            'MetricStat': {
                'Metric': {
                    'Namespace': 'ContainerInsights',
                    'MetricName': 'node_filesystem_utilization',
                    'Dimensions': [{'Name': 'ClusterName', 'Value': cluster_name}]
                },
                'Period': 3600,
                'Stat': 'Average'
            }
        },
        {
            'Id': 'nodes',
            'MetricStat': {
                'Metric': {
                    'Namespace': 'ContainerInsights',
                    'MetricName': 'cluster_node_count',
                    'Dimensions': [{'Name': 'ClusterName', 'Value': cluster_name}]
                },
                'Period': 3600,
                'Stat': 'Average'
            }
        }
    ]

    response = cw.get_metric_data(
        MetricDataQueries=queries,
        StartTime=start_time,
        EndTime=end_time,
        ScanBy='TimestampAscending'
    )

    # Convert results to Series
    results = {res['Id']: pd.Series(res['Values'], index=res['Timestamps']) for res in response['MetricDataResults']}
    
    # Build DataFrame
    df = pd.DataFrame({
        'CPU_Usage': results.get('cpu'),
        'Disk_Usage': results.get('disk'),
        'Number_of_Nodes': results.get('nodes')
    })
    
    # Clean up and handle missing time-slices
    df = df.resample('h').mean().ffill().fillna(0)
    df.index.name = 'DateTime'
    
    return df.reset_index()

# Usage for EKS
eks_df = get_eks_training_data(cluster_name='my-production-cluster', days=14)
eks_df.to_csv('eks_lstm_training_data.csv', index=False)