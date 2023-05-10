package main

import (
	"fmt"
	"os"
)

// samples of dataset, 10 pics are saved

func main() {
	pathes := []string{
		"/Users/yerassyl/Documents/th/dataset/binary_masks",
		"/Users/yerassyl/Documents/th/dataset/images",
		"/Users/yerassyl/Documents/th/dataset/train_img",
		"/Users/yerassyl/Documents/th/dataset/train_masks",
		"/Users/yerassyl/Documents/th/dataset/val_imgs",
		"/Users/yerassyl/Documents/th/dataset/val_masks",
		"/Users/yerassyl/Documents/th/saved_images",
	}

	l := 9
	for i := range pathes {
		files, err := os.ReadDir(pathes[i])
		if err != nil {
			fmt.Print(err)
		}

		for j, file := range files {
			if j > l {
				os.Remove(pathes[i] + "/" + file.Name())
			}
		}

	}
}
