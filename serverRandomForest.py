import pickle, numpy as np
import litserve as ls

class RandomForestAPI(ls.LitAPI):
    def setup(self, device):
        with open("model.pkl", "rb") as f:
            self.model = pickle.load(f)

    def decode_request(self, request):
        x = np.asarray(request["input"])
        x = np.expand_dims(x, 0)
        return x

    def predict(self, x):
        return self.model.predict(x)

    def encode_response(self, output):
        return {"class_idx": int(output)}

    # Adicione este método para lidar com CORS
    def handle_request(self, request, response):
        # Permite requisições do seu frontend
        response.headers["Access-Control-Allow-Origin"] = "https://randoforestclient.vercel.app"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        if request.method == "OPTIONS":
            response.status_code = 204
            response.body = b""
            return response
        return super().handle_request(request, response)

if __name__ == "__main__":
    api = RandomForestAPI()
    server = ls.LitServer(api)
    server.run(port=8000)