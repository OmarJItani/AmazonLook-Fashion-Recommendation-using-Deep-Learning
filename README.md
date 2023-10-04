# AmazonLook: Fashion Recommendation using Deep Learning

Disclaimer: This work is done as part of the Amazon Industry Program 2023.

## Intorduction

Understanding the relationship between fashion items based on their visual characteristics is fundamental to making informed and satisfying shopping choices. People often seek not just alternatives but complementary items that enhance their existing wardrobe pieces. However, the vast array of options available online can overwhelm shoppers, leading to a less satisfying shopping experience.

## What is in this repository?

In this project, we've tackled this issue by developing a tool that models the inherent human sense of relationship between objects based on appearance. Our solution simplifies the online shopping experience by helping users find, in a single click, a complementary fashion item that matches an item they already own. This repository contains the codebase and resources for AmazonLook, a platform designed to enhance the online shopping journey for users.

## Solution
AmazonLook employs cutting-edge artificial intelligence technologies to analyze the style of a fashion item provided by the user and explores its database to find the best matching item available.

Specifically, the system utilizes the ResNet50 convolutional neural network to extract features from the user-provided product. These features are then compared to the features of all items in AmazonLook's database using an L2 Euclidean norm similarity metric. The most similar item, i.e., the item with the least cost, is then recommended to the user.

## How it works?

- Upload your product image: Snap a picture
of the item you already own - it could be a shirt, a dress, or any other clothing item.
- Specify the category you are looking for - shorts, shoes, hats, or any other category.
- We analyze your item and suggest complementary products.

## Website Demonstration

<div style="text-align: center;">
  <img src="Readme_Files/demo_video.gif" alt="AmazonLook" width="500" loop="true"/></a>
</div>

## Getting Started

- Clone the repository on your local machine.
- On your machine, open a terminal and navigate to the cloned directory.
- In the terminal, run:
    ```bash
    python3 app.py
    ```
- In your browser, go to: 
    ```bash
    http://127.0.0.1:5000/
    ```
- Enjoy!