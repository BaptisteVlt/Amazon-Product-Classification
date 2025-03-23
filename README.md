# Amazon-Product-Classification

## How to run the application
Clone the repo.  
Add the amz_products_small.jsonl in your directory.  

Build the container to train the model :  
docker build -t product-category-train -f Dockerfile.train .  

Run the training :  
docker run -v "$(pwd)/amz_products_small.jsonl:/app/amz_products_small.jsonl" --rm product-category-train  

Build the container for the api :  
docker build -t product-category-api -f Dockerfile.api .  

Run the api :  
docker run --rm -p 8000:8000 product-category-api  

Questions :  
To predict more categories  we should train a multi-label classification model instead of a multi-class classification.  
We should have a model that output probability for each label.  

To deploy the api in the cloud we should first choose a cloud provider like AWS. Then push the container image to Amazon ECR.  
Then we will need to monitor the performance of the model overtime.  

If we don't have any label to monitor the model then we should consider checking the distribution of predicted categories.  
