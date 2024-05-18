{
  "nbformat": 4,
  "nbformat_minor": 0,

    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/ljnmuyrong/Final-Examination/blob/main/streamlit_app.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install streamlit"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "mvlSG3yuqca9",
        "outputId": "c287130a-f3c4-4bfe-f9f1-c77ea314f75d"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting streamlit\n",
            "  Downloading streamlit-1.34.0-py2.py3-none-any.whl (8.5 MB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m8.5/8.5 MB\u001b[0m \u001b[31m21.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: altair<6,>=4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.2.2)\n",
            "Requirement already satisfied: blinker<2,>=1.0.0 in /usr/lib/python3/dist-packages (from streamlit) (1.4)\n",
            "Requirement already satisfied: cachetools<6,>=4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (5.3.3)\n",
            "Requirement already satisfied: click<9,>=7.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (8.1.7)\n",
            "Requirement already satisfied: numpy<2,>=1.19.3 in /usr/local/lib/python3.10/dist-packages (from streamlit) (1.25.2)\n",
            "Requirement already satisfied: packaging<25,>=16.8 in /usr/local/lib/python3.10/dist-packages (from streamlit) (24.0)\n",
            "Requirement already satisfied: pandas<3,>=1.3.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (2.0.3)\n",
            "Requirement already satisfied: pillow<11,>=7.1.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (9.4.0)\n",
            "Requirement already satisfied: protobuf<5,>=3.20 in /usr/local/lib/python3.10/dist-packages (from streamlit) (3.20.3)\n",
            "Requirement already satisfied: pyarrow>=7.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (14.0.2)\n",
            "Requirement already satisfied: requests<3,>=2.27 in /usr/local/lib/python3.10/dist-packages (from streamlit) (2.31.0)\n",
            "Requirement already satisfied: rich<14,>=10.14.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (13.7.1)\n",
            "Requirement already satisfied: tenacity<9,>=8.1.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (8.3.0)\n",
            "Requirement already satisfied: toml<2,>=0.10.1 in /usr/local/lib/python3.10/dist-packages (from streamlit) (0.10.2)\n",
            "Requirement already satisfied: typing-extensions<5,>=4.3.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.11.0)\n",
            "Collecting gitpython!=3.1.19,<4,>=3.0.7 (from streamlit)\n",
            "  Downloading GitPython-3.1.43-py3-none-any.whl (207 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m207.3/207.3 kB\u001b[0m \u001b[31m20.8 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hCollecting pydeck<1,>=0.8.0b4 (from streamlit)\n",
            "  Downloading pydeck-0.9.1-py2.py3-none-any.whl (6.9 MB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m6.9/6.9 MB\u001b[0m \u001b[31m62.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: tornado<7,>=6.0.3 in /usr/local/lib/python3.10/dist-packages (from streamlit) (6.3.3)\n",
            "Collecting watchdog>=2.1.5 (from streamlit)\n",
            "  Downloading watchdog-4.0.0-py3-none-manylinux2014_x86_64.whl (82 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m83.0/83.0 kB\u001b[0m \u001b[31m8.6 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: entrypoints in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (0.4)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (3.1.4)\n",
            "Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (4.19.2)\n",
            "Requirement already satisfied: toolz in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (0.12.1)\n",
            "Collecting gitdb<5,>=4.0.1 (from gitpython!=3.1.19,<4,>=3.0.7->streamlit)\n",
            "  Downloading gitdb-4.0.11-py3-none-any.whl (62 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m62.7/62.7 kB\u001b[0m \u001b[31m6.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=1.3.0->streamlit) (2.8.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=1.3.0->streamlit) (2023.4)\n",
            "Requirement already satisfied: tzdata>=2022.1 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=1.3.0->streamlit) (2024.1)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (3.3.2)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (3.7)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (2.0.7)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (2024.2.2)\n",
            "Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich<14,>=10.14.0->streamlit) (3.0.0)\n",
            "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich<14,>=10.14.0->streamlit) (2.16.1)\n",
            "Collecting smmap<6,>=3.0.1 (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit)\n",
            "  Downloading smmap-5.0.1-py3-none-any.whl (24 kB)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->altair<6,>=4.0->streamlit) (2.1.5)\n",
            "Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (23.2.0)\n",
            "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (2023.12.1)\n",
            "Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.35.1)\n",
            "Requirement already satisfied: rpds-py>=0.7.1 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.18.1)\n",
            "Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich<14,>=10.14.0->streamlit) (0.1.2)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->pandas<3,>=1.3.0->streamlit) (1.16.0)\n",
            "Installing collected packages: watchdog, smmap, pydeck, gitdb, gitpython, streamlit\n",
            "Successfully installed gitdb-4.0.11 gitpython-3.1.43 pydeck-0.9.1 smmap-5.0.1 streamlit-1.34.0 watchdog-4.0.0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install pyngrok"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8ox0qlzlqg2b",
        "outputId": "4498515c-c97c-46f7-db69-cb0f15b9ebeb"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting pyngrok\n",
            "  Downloading pyngrok-7.1.6-py3-none-any.whl (22 kB)\n",
            "Requirement already satisfied: PyYAML>=5.1 in /usr/local/lib/python3.10/dist-packages (from pyngrok) (6.0.1)\n",
            "Installing collected packages: pyngrok\n",
            "Successfully installed pyngrok-7.1.6\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8EVEjZBNp5Pj",
        "outputId": "37bb8e66-4ab9-42b0-f3eb-896071e861b4"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Writing app.py\n"
          ]
        }
      ],
      "source": [
        "%%writefile app.py"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "%%writefile requirements.txt"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "D02oiBEZqBrt",
        "outputId": "78c9f7af-1853-4cd3-a1ed-29c7885027ab"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Writing requirements.txt\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import streamlit as st\n",
        "import requests\n",
        "import h5py\n",
        "import tensorflow as tf\n",
        "from io import BytesIO\n",
        "from PIL import Image"
      ],
      "metadata": {
        "id": "dZ7aEsqrqTtT"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Streamlit app\n",
        "st.title(\"Image Classifier\")\n",
        "\n",
        "uploaded_file = st.file_uploader(\"Choose an image...\", type=\"png\")\n",
        "\n",
        "if uploaded_file is not None:\n",
        "    # Load the image\n",
        "    img = Image.open(uploaded_file).convert('L')\n",
        "    img = img.resize((28, 28))\n",
        "    img_array = np.array(img)\n",
        "    img_array = img_array / 255.0\n",
        "    img_array = img_array.reshape(1, 28, 28, 1)\n",
        "\n",
        "    # Display the image\n",
        "    st.image(img, caption='Uploaded Image', use_column_width=True)\n",
        "    st.write(\"\")\n",
        "    st.write(\"Classifying...\")\n",
        "\n",
        "    # Predict the class\n",
        "    prediction = loaded_model.predict(img_array)\n",
        "    st.write(f\"Predicted Class: {np.argmax(prediction)}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "nWpZiNoWqD7k",
        "outputId": "f555b7ce-075c-4347-ac1f-3e39da71d2ac"
      },
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2024-05-18 07:46:52.816 \n",
            "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
            "  command:\n",
            "\n",
            "    streamlit run /usr/local/lib/python3.10/dist-packages/colab_kernel_launcher.py [ARGUMENTS]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "class_mapping = {\n",
        "    0: 'puppy',\n",
        "    1: 'airplane',\n",
        "    2: 'Dinosaur',\n",
        "    3: 'flag',\n",
        "}\n",
        "\n",
        "@st.cache(allow_output_mutation=True)\n",
        "def load_model():\n",
        "    base_url = \"https://github.com/ljnmuyrong/Final-Examination/raw/main/\"\n",
        "    model_parts = [f\"{base_url}bestmodel.hdf5.h5.part{i:02d}\" for i in range(1, 35)]\n",
        "\n",
        "    model_bytes = b''\n",
        "    for part_url in model_parts:\n",
        "        response = requests.get(part_url)\n",
        "        model_bytes += response.content\n",
        "\n",
        "    with h5py.File(BytesIO(model_bytes), 'r') as hf:\n",
        "        model = tf.keras.models.load_model(hf)\n",
        "\n",
        "    return model"
      ],
      "metadata": {
        "id": "a5AT6i4xq28I"
      },
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def predict(image, model):\n",
        "    img_array = np.array(image)\n",
        "    img_array = tf.image.resize(img_array, (150, 150))\n",
        "    img_array = tf.expand_dims(img_array, 0)\n",
        "    img_array = img_array / 255.0\n",
        "\n",
        "    predictions = model.predict(img_array)\n",
        "    predicted_class = class_mapping[np.argmax(predictions[0])]\n",
        "    return predicted_class"
      ],
      "metadata": {
        "id": "5CtjIhPqq48F"
      },
      "execution_count": 11,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "st.title('Imagee Classification')\n",
        "uploaded_file = st.file_uploader(\"Choose an image\", type=[\"jpg\", \"jpeg\", \"png\"])\n",
        "\n",
        "if uploaded_file is not None:\n",
        "    image = Image.open(uploaded_file)\n",
        "    st.image(image, caption='Uploaded Image.', use_column_width=True)\n",
        "    model = load_model()\n",
        "\n",
        "    predicted_class = predict(image, model)\n",
        "    st.write(f\"Prediction: {predicted_class}\")"
      ],
      "metadata": {
        "id": "B37luneiq6I7"
      },
      "execution_count": 12,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!streamlit run /content/drive/MyDrive/Data/app.py&>/dev/null&"
      ],
      "metadata": {
        "id": "Yunj3kKYq8Le"
      },
      "execution_count": 13,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!ngrok authtoken 2gcsXIj9AYMniXnkaJLyBjaBgm3_5Kz6QrCZK5euyXT95EfZK"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "N6YbY5dNrCuo",
        "outputId": "d933939b-7901-4845-ebb5-c90af2515fa6"
      },
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Authtoken saved to configuration file: /root/.config/ngrok/ngrok.yml\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from pyngrok import ngrok\n",
        "\n",
        "public_url = ngrok.connect(addr='8501')\n",
        "print(f\"Streamlit URL: {public_url}\")\n",
        "\n",
        "!streamlit run /content/drive/MyDrive/Data/app.py &>/dev/null&\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "RWWJZSoHrEI2",
        "outputId": "d2d4b8c8-6005-474b-933f-8b50d96ac5a4"
      },
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Streamlit URL: NgrokTunnel: \"https://5eed-34-147-61-35.ngrok-free.app\" -> \"http://localhost:8501\"\n"
          ]
        }
      ]
    }
  ]
}
