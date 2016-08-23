from evaluate import strict, loose_macro, loose_micro

def loss_hook(batcher,loss_model,x_context,x_target,y,keep_prob_context,keep_prob_target,context_length,sess):
    [x_context_data, x_target_mean_data, y_data] = batcher.next()
    num_of_samples = y_data.shape[0]
    feed = {y:y_data,keep_prob_context:[1.],keep_prob_target:[1.]}
    for i in range(context_length*2+1):
        feed[x_context[i]] = x_context_data[:,i,:]
    feed[x_target] = x_target_mean_data
    loss = sess.run(loss_model, feed_dict = feed)
    print("loss:", loss / num_of_samples)


def acc_hook(batcher,output_model,x_context,x_target,feature,y,keep_prob_context,keep_prob_target,context_length,sess):
    true_and_prediction = []
    [x_context_data, x_target_mean_data, y_data,feature_data] = batcher.next()
    num_of_samples = y_data.shape[0]
    feed = {y:y_data,keep_prob_context:[1.],keep_prob_target:[1.],feature:feature_data}
    for i in range(context_length*2+1):
        feed[x_context[i]] = x_context_data[:,i,:]
    feed[x_target] = x_target_mean_data

    scores = sess.run(output_model, feed_dict = feed)
    """
    for score,true_label in zip(scores,y_data):
        predicted_tag = []
        true_tag = []
        for label_id,label_score in enumerate(list(true_label)):
            if label_score > 0:
                true_tag.append(label_id)
        for label_id,label_score in enumerate(list(score)):
            if label_score > 0.5:
                predicted_tag.append(label_id)
        true_and_prediction.append((true_tag, predicted_tag))
    """    
    for score,true_label in zip(scores,y_data):
        predicted_tag = []                                                                                                                                                                                 
        true_tag = []
        for label_id,label_score in enumerate(list(true_label)):
            if label_score > 0:
                true_tag.append(label_id)
        lid,ls = max(enumerate(list(score)),key=lambda x: x[1])
        predicted_tag.append(lid)
        for label_id,label_score in enumerate(list(score)):
            if label_score > 0.5:
                if label_id != lid:
                    predicted_tag.append(label_id)
        true_and_prediction.append((true_tag, predicted_tag))
    
    print("     strict (p,r,f1):",strict(true_and_prediction))
    print("loose macro (p,r,f1):",loose_macro(true_and_prediction))
    print("loose micro (p,r,f1):",loose_micro(true_and_prediction))






