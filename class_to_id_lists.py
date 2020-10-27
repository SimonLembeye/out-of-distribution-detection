def build_cifar_10_class_to_id_list_10():
    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truc"]
    cifar_10_class_to_id_list_10 = []
    for i in range(len(classes)):
        dict = {}
        p = 0
        for j in range(len(classes)):
            if i != j:
                dict[classes[j]] = p
                p += 1

        cifar_10_class_to_id_list_10.append(dict)

    return cifar_10_class_to_id_list_10



cifar_10_class_to_id_list_5 = [
    {
        "airplane": 0,
        "automobile": 1,
        "bird": 2,
        "cat": 3,
        "deer": 4,
        "dog": 5,
        "frog": 6,
        "horse": 7,
    },
    {
        "airplane": 0,
        "automobile": 1,
        "bird": 2,
        "cat": 3,
        "deer": 4,
        "dog": 5,
        "ship": 6,
        "truck": 7,
    },
    {
        "airplane": 0,
        "automobile": 1,
        "bird": 2,
        "cat": 3,
        "frog": 4,
        "horse": 5,
        "ship": 6,
        "truck": 7,
    },
    {
        "airplane": 0,
        "automobile": 1,
        "deer": 2,
        "dog": 3,
        "frog": 4,
        "horse": 5,
        "ship": 6,
        "truck": 7,
    },
    {
        "bird": 0,
        "cat": 1,
        "deer": 2,
        "dog": 3,
        "frog": 4,
        "horse": 5,
        "ship": 6,
        "truck": 7,
    },
]