import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path', type=str)
    parser.add_argument('--labels_path', type=str)
    parser.add_argument('--output_annotation_path', type=str)
    parser.add_argument('--output_labels_path', type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    labels = pd.read_csv(args.labels_path)
    is_tag = labels['attribute_name'].apply(lambda x: 'tag' in x)
    labels.loc[is_tag, 'sub_attribute_id'] = range(sum(is_tag))
    labels.loc[~is_tag, 'sub_attribute_id'] = range(sum(~is_tag))
    tag_ids = set(labels.loc[is_tag, 'attribute_id'])
    culture_ids = set(labels.loc[~is_tag, 'attribute_id'])
    labels['is_tag'] = is_tag

    id2sub_id = dict(zip(labels['attribute_id'], labels['sub_attribute_id']))

    annotation = pd.read_csv(args.annotation_path, converters={'attribute_ids': lambda x: list(map(int, x.split()))})
    annotation['tag_ids'] = annotation['attribute_ids'].apply(lambda x: list(set(x) & tag_ids))
    annotation['culture_ids'] = annotation['attribute_ids'].apply(lambda x: list(set(x) & culture_ids))
    annotation['sub_tag_ids'] = annotation['tag_ids'].apply(lambda x: [id2sub_id[y] for y in x])
    annotation['sub_culture_ids'] = annotation['culture_ids'].apply(lambda x: [id2sub_id[y] for y in x])
    annotation['attribute_ids'] = annotation['attribute_ids']

    annotation.to_csv(args.output_annotation_path, index=False)
    labels.to_csv(args.output_labels_path, index=False)


if __name__ == '__main__':
    main()
