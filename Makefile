
run2:
	fastapi dev --port 8890 --host 0.0.0.0 sm2/app.py

run:
	python src/solvmate/app/app.py 

open:
	open "https://127.0.0.1:8890"

all_fast:
	make clean 
	make training_data
	make pairs
	make train_recommender

all_slow:
	make clean 
	make download_data 
	make xtb_features 
	make training_data
	make pairs
	make train_recommender

clean:
	make clean_training_data 
	make clean_pairs  
	make clean_recommender 


download_data:
	python scripts/download_data.py

clean_pairs:
	rm "data/pairs.db" || echo "skipping clean pairs" 

clean_training_data:
	rm "data/training_data.db" || echo "skipping clean training data" 

clean_recommender:
	rm "data/recommender.pkl" || echo "skipping clean recommender"


pairs:
	python scripts/build_pairs.py

training_data:
	python scripts/build_training_data.py

xtb_features:
	python scripts/build_xtb_features.py

train_recommender:
	python scripts/train_recommender.py --finalretrain

hyperp_recommender:
	python scripts/train_recommender.py --hyperp

smi_to_name:
	python scripts/build_smi_to_name.py 

delete_solvent_selection_db:
	rm data/solvent_selection.db

cert_https:
	cd cert && openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 365 -nodes

screen_model_types:
	python scripts/train_recommender.py --hyperp --screen-model-types --job-name screen_model_types


docker_build:
	docker build --build-arg HTTP_PROXY=$$HTTP_PROXY --build-arg HTTPS_PROXY=$$HTTPS_PROXY --build-arg http_proxy=$$http_proxy --build-arg https_proxy=$$https_proxy -t solvmate:latest .

docker_run:
	(docker rm "/solvmate-server" || true) && docker run --name solvmate-server -p 8890:8890 solvmate:latest


paper:
	python scripts/train_recommender.py --paper