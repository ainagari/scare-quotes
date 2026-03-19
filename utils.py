import json
import re



# Configuration: annotation levels to analyze
HIGH_LEVEL = "quote-type-broad"  # Broad classification (SQ, NSQ, etc.)
FINE_LEVEL = "quote-type-fine1"  # Fine-grained subcategories

def get_all_labels(item, separated=False,ignore_unsure_ambiguous=False):
    """
    Extract both broad and fine-grained labels from an annotated item.
    
    Args:
        item: Annotated corpus item containing annotation results
        separated: If True, return dict with 'high' and 'fine' keys; 
                  if False, return tuple of all labels
    
    Returns:
        Dictionary or tuple of labels depending on 'separated' parameter
    """
    broad_labels = get_labels_one_level(item, level='quote-type-broad', ignore_unsure_ambiguous=ignore_unsure_ambiguous)
    fine_labels = get_labels_one_level(item, level='quote-type-fine1', ignore_unsure_ambiguous=ignore_unsure_ambiguous)
    
    if separated:
        return {'high': broad_labels, 'fine': fine_labels}
    else:
        return tuple(broad_labels + fine_labels)


def get_labels_one_level(item, level, ignore_unsure_ambiguous=False):
    """
    Extract labels from a specific annotation level.
    
    Args:
        item: Annotated corpus item
        level: Annotation level name ('quote-type-broad' or 'quote-type-fine1')
    
    Returns:
        Tuple of label strings for the specified level
    """
    labels = []
    for result in item['annotations'][0]['result']:
        if result['from_name'] == level:
            labels = result['value']['choices'] 
    if ignore_unsure_ambiguous:
    	new_labels = [lb for lb in labels if "Unsure" not in lb and "Ambiguous" not in lb]
    else:
        new_labels = [lb for lb in labels]
    
    return tuple(new_labels)
    


def get_conv_id(utt_id):
    """
    Extract conversation ID from utterance ID.
    
    Args:
        utt_id: Full utterance ID string
    
    Returns:
        Conversation ID (first two components of utterance ID)
    """
    conv_id = "_".join(utt_id.split("_")[:2])
    return conv_id
