def paragraph_sim(para1,para2):
    list1=para1.split(".")
    list2=para2.split(".")
    deflt=60
    i=0
    j=0
    summ=0
    for i in range(len(list1)):
        maxm=0
        for j in range(len(list2)):
            a=sim(list1[i],list2[j]) #calling/importing the similarity checker part. Function of the final_model file
            if(a>maxm):
                maxm=a
        summ+=maxm
    return summ/len(list1)
#Answer by thresholding summ variable


#Having scoring between -.5 and .5 will penalize dissimilarities and hence would avoid wrong classifications. 
#Also, we may have a method where pick the top 2 scorers for a particular sentence and average them out to get a score. 
#def sim(sentence1,sentence2):
    #The trained neural network for sentence similarity checker.
    #The neural net should take the actual English sentence as it's input and give it's output as the similarity score.
#A[]=0
#for i in range(len(list1)):
# for j in range(len(list2)):
#   A[i,j]=sim(list1[i],list2[j])
#Having scoring between -.5 and .5 will penalize dissimilarities and hence would avoid wrong classifications.   
# Similarity score between a sentence of first para and the second para=Avg of that particular sentence's similarity scores.
# Then we will get a paragragh similarity vector.
# Then we can have a condition like if three of them are greater than 0, then we may say that the paragraphs are similar.
#
#
#
#
#
#
#

    
