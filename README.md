# asterisky

Notes on [**asterism**](https://asterism.neetocode.com/site/nidhi-singh-rathore/01JZP27CXXVNVJSP67HKQCJMKD)

# Affinities
This is a thought experiment/sketch to extend **Asterism** such that image boards can be be iterated along three axes:
1. **formal affinities** - morphology, color, medium/technology
2. **conceptual affinities** - subject matter, intentions, associations, influences
3. **historical affinities** - date of creation, provenance, contemporary events

There may be a better set of dimensions for this problem, but this is a simple, small number for illustration.

We propose using generative AI to build a graph udnerlying a recommendation system.

A graph of all images from the selected databases that maps them into a space then allows identification of where a specific image falls relative to all others, and 'recommend' images of varying (user-specified?) adjacency on an axis from nearest neighbor to most distant.

This lays the groundwork for a version of **Asterism** as proposed to Nidhi and Marc via email correspondence:

> For each image, I want a rich description of what it is AND a way to connect it to, say 5 formally or conceptually 'adjacent' images - then user can either replace the current image with an adjacent image OR replace another image on screen SUCH that a user could build a collection of images that explores a thread of an idea.
>
> I'm imagining a kind of visual gin rummy.
> + Player is dealt 7 cards (images).
> + Each card when flipped over (from image on the face) reveals 5 more possible cards (5 more images, or even 3) that are associated to the card by formal or conceptual similarities.
> + Player can elect to do nothing, replace current card with a possible card, or replace another card in the hand.
> + so ultimately, the player can build an 'ideal' hand - seven cards (images) that started as random but were refined by exploring an exciting image from the initial deal.

# Additional app features for consideration
+ add scaling
+ add Z-position
+ add rotation
+ add citation (copy to clipboard)
+ add image download (best res)
+ add export board to .PDF
+ add HSB controls
+ add alpha
