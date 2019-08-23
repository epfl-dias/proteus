{
  "operator" : "print",
  "gpu" : false,
  "e" : [ {
    "expression" : "recordProjection",
    "e" : {
      "expression" : "argument",
      "attributes" : [ {
        "attrName" : "d_year",
        "relName" : "subsetPelagoSort#5489"
      } ],
      "type" : {
        "relName" : "subsetPelagoSort#5489",
        "type" : "record"
      },
      "argNo" : -1
    },
    "attribute" : {
      "attrName" : "d_year",
      "relName" : "subsetPelagoSort#5489"
    },
    "register_as" : {
      "attrName" : "d_year",
      "relName" : "print5490"
    }
  }, {
    "expression" : "recordProjection",
    "e" : {
      "expression" : "argument",
      "attributes" : [ {
        "attrName" : "s_city",
        "relName" : "subsetPelagoSort#5489"
      } ],
      "type" : {
        "relName" : "subsetPelagoSort#5489",
        "type" : "record"
      },
      "argNo" : -1
    },
    "attribute" : {
      "attrName" : "s_city",
      "relName" : "subsetPelagoSort#5489"
    },
    "register_as" : {
      "attrName" : "s_city",
      "relName" : "print5490"
    }
  }, {
    "expression" : "recordProjection",
    "e" : {
      "expression" : "argument",
      "attributes" : [ {
        "attrName" : "p_brand1",
        "relName" : "subsetPelagoSort#5489"
      } ],
      "type" : {
        "relName" : "subsetPelagoSort#5489",
        "type" : "record"
      },
      "argNo" : -1
    },
    "attribute" : {
      "attrName" : "p_brand1",
      "relName" : "subsetPelagoSort#5489"
    },
    "register_as" : {
      "attrName" : "p_brand1",
      "relName" : "print5490"
    }
  }, {
    "expression" : "recordProjection",
    "e" : {
      "expression" : "argument",
      "attributes" : [ {
        "attrName" : "profit",
        "relName" : "subsetPelagoSort#5489"
      } ],
      "type" : {
        "relName" : "subsetPelagoSort#5489",
        "type" : "record"
      },
      "argNo" : -1
    },
    "attribute" : {
      "attrName" : "profit",
      "relName" : "subsetPelagoSort#5489"
    },
    "register_as" : {
      "attrName" : "profit",
      "relName" : "print5490"
    }
  } ],
  "input" : {
    "operator" : "project",
    "gpu" : false,
    "e" : [ {
      "e" : {
        "e" : {
          "attributes" : [ {
            "attrName" : "__sorted",
            "relName" : "__sort5489"
          } ],
          "expression" : "argument",
          "type" : {
            "type" : "record",
            "relName" : "__sort5489"
          },
          "argNo" : 1
        },
        "expression" : "recordProjection",
        "attribute" : {
          "attrName" : "__sorted",
          "relName" : "__sort5489"
        }
      },
      "expression" : "recordProjection",
      "attribute" : {
        "attrName" : "d_year",
        "relName" : "__sort5489"
      },
      "register_as" : {
        "attrName" : "d_year",
        "relName" : "subsetPelagoSort#5489"
      }
    }, {
      "e" : {
        "e" : {
          "attributes" : [ {
            "attrName" : "__sorted",
            "relName" : "__sort5489"
          } ],
          "expression" : "argument",
          "type" : {
            "type" : "record",
            "relName" : "__sort5489"
          },
          "argNo" : 1
        },
        "expression" : "recordProjection",
        "attribute" : {
          "attrName" : "__sorted",
          "relName" : "__sort5489"
        }
      },
      "expression" : "recordProjection",
      "attribute" : {
        "attrName" : "s_city",
        "relName" : "__sort5489"
      },
      "register_as" : {
        "attrName" : "s_city",
        "relName" : "subsetPelagoSort#5489"
      }
    }, {
      "e" : {
        "e" : {
          "attributes" : [ {
            "attrName" : "__sorted",
            "relName" : "__sort5489"
          } ],
          "expression" : "argument",
          "type" : {
            "type" : "record",
            "relName" : "__sort5489"
          },
          "argNo" : 1
        },
        "expression" : "recordProjection",
        "attribute" : {
          "attrName" : "__sorted",
          "relName" : "__sort5489"
        }
      },
      "expression" : "recordProjection",
      "attribute" : {
        "attrName" : "p_brand1",
        "relName" : "__sort5489"
      },
      "register_as" : {
        "attrName" : "p_brand1",
        "relName" : "subsetPelagoSort#5489"
      }
    }, {
      "e" : {
        "e" : {
          "attributes" : [ {
            "attrName" : "__sorted",
            "relName" : "__sort5489"
          } ],
          "expression" : "argument",
          "type" : {
            "type" : "record",
            "relName" : "__sort5489"
          },
          "argNo" : 1
        },
        "expression" : "recordProjection",
        "attribute" : {
          "attrName" : "__sorted",
          "relName" : "__sort5489"
        }
      },
      "expression" : "recordProjection",
      "attribute" : {
        "attrName" : "profit",
        "relName" : "__sort5489"
      },
      "register_as" : {
        "attrName" : "profit",
        "relName" : "subsetPelagoSort#5489"
      }
    } ],
    "relName" : "subsetPelagoSort#5489",
    "input" : {
      "operator" : "unpack",
      "gpu" : false,
      "projections" : [ {
        "e" : {
          "attributes" : [ {
            "attrName" : "__sorted",
            "relName" : "__sort5489"
          } ],
          "expression" : "argument",
          "type" : {
            "type" : "record",
            "relName" : "__sort5489"
          },
          "argNo" : 1
        },
        "expression" : "recordProjection",
        "attribute" : {
          "attrName" : "__sorted",
          "relName" : "__sort5489"
        }
      } ],
      "input" : {
        "operator" : "sort",
        "gpu" : false,
        "rowType" : [ {
          "relName" : "__sort5489",
          "attrName" : "d_year"
        }, {
          "relName" : "__sort5489",
          "attrName" : "s_city"
        }, {
          "relName" : "__sort5489",
          "attrName" : "p_brand1"
        }, {
          "relName" : "__sort5489",
          "attrName" : "profit"
        } ],
        "e" : [ {
          "direction" : "ASC",
          "expression" : {
            "expression" : "recordProjection",
            "e" : {
              "expression" : "argument",
              "attributes" : [ {
                "attrName" : "d_year",
                "relName" : "subsetPelagoUnpack#5488"
              } ],
              "type" : {
                "relName" : "subsetPelagoUnpack#5488",
                "type" : "record"
              },
              "argNo" : -1
            },
            "attribute" : {
              "attrName" : "d_year",
              "relName" : "subsetPelagoUnpack#5488"
            },
            "register_as" : {
              "attrName" : "d_year",
              "relName" : "__sort5489"
            }
          }
        }, {
          "direction" : "ASC",
          "expression" : {
            "expression" : "recordProjection",
            "e" : {
              "expression" : "argument",
              "attributes" : [ {
                "attrName" : "s_city",
                "relName" : "subsetPelagoUnpack#5488"
              } ],
              "type" : {
                "relName" : "subsetPelagoUnpack#5488",
                "type" : "record"
              },
              "argNo" : -1
            },
            "attribute" : {
              "attrName" : "s_city",
              "relName" : "subsetPelagoUnpack#5488"
            },
            "register_as" : {
              "attrName" : "s_city",
              "relName" : "__sort5489"
            }
          }
        }, {
          "direction" : "ASC",
          "expression" : {
            "expression" : "recordProjection",
            "e" : {
              "expression" : "argument",
              "attributes" : [ {
                "attrName" : "p_brand1",
                "relName" : "subsetPelagoUnpack#5488"
              } ],
              "type" : {
                "relName" : "subsetPelagoUnpack#5488",
                "type" : "record"
              },
              "argNo" : -1
            },
            "attribute" : {
              "attrName" : "p_brand1",
              "relName" : "subsetPelagoUnpack#5488"
            },
            "register_as" : {
              "attrName" : "p_brand1",
              "relName" : "__sort5489"
            }
          }
        }, {
          "direction" : "NONE",
          "expression" : {
            "expression" : "recordProjection",
            "e" : {
              "expression" : "argument",
              "attributes" : [ {
                "attrName" : "profit",
                "relName" : "subsetPelagoUnpack#5488"
              } ],
              "type" : {
                "relName" : "subsetPelagoUnpack#5488",
                "type" : "record"
              },
              "argNo" : -1
            },
            "attribute" : {
              "attrName" : "profit",
              "relName" : "subsetPelagoUnpack#5488"
            },
            "register_as" : {
              "attrName" : "profit",
              "relName" : "__sort5489"
            }
          }
        } ],
        "granularity" : "thread",
        "input" : {
          "operator" : "unpack",
          "gpu" : false,
          "projections" : [ {
            "expression" : "recordProjection",
            "e" : {
              "expression" : "argument",
              "attributes" : [ {
                "attrName" : "d_year",
                "relName" : "subsetPelagoUnpack#5488"
              } ],
              "type" : {
                "relName" : "subsetPelagoUnpack#5488",
                "type" : "record"
              },
              "argNo" : -1
            },
            "attribute" : {
              "attrName" : "d_year",
              "relName" : "subsetPelagoUnpack#5488"
            }
          }, {
            "expression" : "recordProjection",
            "e" : {
              "expression" : "argument",
              "attributes" : [ {
                "attrName" : "s_city",
                "relName" : "subsetPelagoUnpack#5488"
              } ],
              "type" : {
                "relName" : "subsetPelagoUnpack#5488",
                "type" : "record"
              },
              "argNo" : -1
            },
            "attribute" : {
              "attrName" : "s_city",
              "relName" : "subsetPelagoUnpack#5488"
            }
          }, {
            "expression" : "recordProjection",
            "e" : {
              "expression" : "argument",
              "attributes" : [ {
                "attrName" : "p_brand1",
                "relName" : "subsetPelagoUnpack#5488"
              } ],
              "type" : {
                "relName" : "subsetPelagoUnpack#5488",
                "type" : "record"
              },
              "argNo" : -1
            },
            "attribute" : {
              "attrName" : "p_brand1",
              "relName" : "subsetPelagoUnpack#5488"
            }
          }, {
            "expression" : "recordProjection",
            "e" : {
              "expression" : "argument",
              "attributes" : [ {
                "attrName" : "profit",
                "relName" : "subsetPelagoUnpack#5488"
              } ],
              "type" : {
                "relName" : "subsetPelagoUnpack#5488",
                "type" : "record"
              },
              "argNo" : -1
            },
            "attribute" : {
              "attrName" : "profit",
              "relName" : "subsetPelagoUnpack#5488"
            }
          } ],
          "input" : {
            "operator" : "mem-move-device",
            "projections" : [ {
              "relName" : "subsetPelagoUnpack#5488",
              "attrName" : "d_year",
              "isBlock" : true
            }, {
              "relName" : "subsetPelagoUnpack#5488",
              "attrName" : "s_city",
              "isBlock" : true
            }, {
              "relName" : "subsetPelagoUnpack#5488",
              "attrName" : "p_brand1",
              "isBlock" : true
            }, {
              "relName" : "subsetPelagoUnpack#5488",
              "attrName" : "profit",
              "isBlock" : true
            } ],
            "input" : {
              "operator" : "gpu-to-cpu",
              "projections" : [ {
                "relName" : "subsetPelagoUnpack#5488",
                "attrName" : "d_year",
                "isBlock" : true
              }, {
                "relName" : "subsetPelagoUnpack#5488",
                "attrName" : "s_city",
                "isBlock" : true
              }, {
                "relName" : "subsetPelagoUnpack#5488",
                "attrName" : "p_brand1",
                "isBlock" : true
              }, {
                "relName" : "subsetPelagoUnpack#5488",
                "attrName" : "profit",
                "isBlock" : true
              } ],
              "queueSize" : 262144,
              "granularity" : "thread",
              "input" : {
                "operator" : "pack",
                "gpu" : true,
                "projections" : [ {
                  "expression" : "recordProjection",
                  "e" : {
                    "expression" : "argument",
                    "attributes" : [ {
                      "attrName" : "d_year",
                      "relName" : "subsetPelagoUnpack#5488"
                    } ],
                    "type" : {
                      "relName" : "subsetPelagoUnpack#5488",
                      "type" : "record"
                    },
                    "argNo" : -1
                  },
                  "attribute" : {
                    "attrName" : "d_year",
                    "relName" : "subsetPelagoUnpack#5488"
                  }
                }, {
                  "expression" : "recordProjection",
                  "e" : {
                    "expression" : "argument",
                    "attributes" : [ {
                      "attrName" : "s_city",
                      "relName" : "subsetPelagoUnpack#5488"
                    } ],
                    "type" : {
                      "relName" : "subsetPelagoUnpack#5488",
                      "type" : "record"
                    },
                    "argNo" : -1
                  },
                  "attribute" : {
                    "attrName" : "s_city",
                    "relName" : "subsetPelagoUnpack#5488"
                  }
                }, {
                  "expression" : "recordProjection",
                  "e" : {
                    "expression" : "argument",
                    "attributes" : [ {
                      "attrName" : "p_brand1",
                      "relName" : "subsetPelagoUnpack#5488"
                    } ],
                    "type" : {
                      "relName" : "subsetPelagoUnpack#5488",
                      "type" : "record"
                    },
                    "argNo" : -1
                  },
                  "attribute" : {
                    "attrName" : "p_brand1",
                    "relName" : "subsetPelagoUnpack#5488"
                  }
                }, {
                  "expression" : "recordProjection",
                  "e" : {
                    "expression" : "argument",
                    "attributes" : [ {
                      "attrName" : "profit",
                      "relName" : "subsetPelagoUnpack#5488"
                    } ],
                    "type" : {
                      "relName" : "subsetPelagoUnpack#5488",
                      "type" : "record"
                    },
                    "argNo" : -1
                  },
                  "attribute" : {
                    "attrName" : "profit",
                    "relName" : "subsetPelagoUnpack#5488"
                  }
                } ],
                "input" : {
                  "operator" : "groupby",
                  "gpu" : true,
                  "k" : [ {
                    "expression" : "recordProjection",
                    "e" : {
                      "expression" : "argument",
                      "attributes" : [ {
                        "attrName" : "d_year",
                        "relName" : "subsetPelagoUnpack#5484"
                      } ],
                      "type" : {
                        "relName" : "subsetPelagoUnpack#5484",
                        "type" : "record"
                      },
                      "argNo" : -1
                    },
                    "attribute" : {
                      "attrName" : "d_year",
                      "relName" : "subsetPelagoUnpack#5484"
                    },
                    "register_as" : {
                      "attrName" : "d_year",
                      "relName" : "subsetPelagoUnpack#5488"
                    }
                  }, {
                    "expression" : "recordProjection",
                    "e" : {
                      "expression" : "argument",
                      "attributes" : [ {
                        "attrName" : "s_city",
                        "relName" : "subsetPelagoUnpack#5484"
                      } ],
                      "type" : {
                        "relName" : "subsetPelagoUnpack#5484",
                        "type" : "record"
                      },
                      "argNo" : -1
                    },
                    "attribute" : {
                      "attrName" : "s_city",
                      "relName" : "subsetPelagoUnpack#5484"
                    },
                    "register_as" : {
                      "attrName" : "s_city",
                      "relName" : "subsetPelagoUnpack#5488"
                    }
                  }, {
                    "expression" : "recordProjection",
                    "e" : {
                      "expression" : "argument",
                      "attributes" : [ {
                        "attrName" : "p_brand1",
                        "relName" : "subsetPelagoUnpack#5484"
                      } ],
                      "type" : {
                        "relName" : "subsetPelagoUnpack#5484",
                        "type" : "record"
                      },
                      "argNo" : -1
                    },
                    "attribute" : {
                      "attrName" : "p_brand1",
                      "relName" : "subsetPelagoUnpack#5484"
                    },
                    "register_as" : {
                      "attrName" : "p_brand1",
                      "relName" : "subsetPelagoUnpack#5488"
                    }
                  } ],
                  "e" : [ {
                    "m" : "sum",
                    "e" : {
                      "expression" : "recordProjection",
                      "e" : {
                        "expression" : "argument",
                        "attributes" : [ {
                          "attrName" : "profit",
                          "relName" : "subsetPelagoUnpack#5484"
                        } ],
                        "type" : {
                          "relName" : "subsetPelagoUnpack#5484",
                          "type" : "record"
                        },
                        "argNo" : -1
                      },
                      "attribute" : {
                        "attrName" : "profit",
                        "relName" : "subsetPelagoUnpack#5484"
                      },
                      "register_as" : {
                        "attrName" : "profit",
                        "relName" : "subsetPelagoUnpack#5488"
                      }
                    },
                    "packet" : 1,
                    "offset" : 0
                  } ],
                  "hash_bits" : 10,
                  "maxInputSize" : 131072,
                  "input" : {
                    "operator" : "unpack",
                    "gpu" : true,
                    "projections" : [ {
                      "expression" : "recordProjection",
                      "e" : {
                        "expression" : "argument",
                        "attributes" : [ {
                          "attrName" : "d_year",
                          "relName" : "subsetPelagoUnpack#5484"
                        } ],
                        "type" : {
                          "relName" : "subsetPelagoUnpack#5484",
                          "type" : "record"
                        },
                        "argNo" : -1
                      },
                      "attribute" : {
                        "attrName" : "d_year",
                        "relName" : "subsetPelagoUnpack#5484"
                      }
                    }, {
                      "expression" : "recordProjection",
                      "e" : {
                        "expression" : "argument",
                        "attributes" : [ {
                          "attrName" : "s_city",
                          "relName" : "subsetPelagoUnpack#5484"
                        } ],
                        "type" : {
                          "relName" : "subsetPelagoUnpack#5484",
                          "type" : "record"
                        },
                        "argNo" : -1
                      },
                      "attribute" : {
                        "attrName" : "s_city",
                        "relName" : "subsetPelagoUnpack#5484"
                      }
                    }, {
                      "expression" : "recordProjection",
                      "e" : {
                        "expression" : "argument",
                        "attributes" : [ {
                          "attrName" : "p_brand1",
                          "relName" : "subsetPelagoUnpack#5484"
                        } ],
                        "type" : {
                          "relName" : "subsetPelagoUnpack#5484",
                          "type" : "record"
                        },
                        "argNo" : -1
                      },
                      "attribute" : {
                        "attrName" : "p_brand1",
                        "relName" : "subsetPelagoUnpack#5484"
                      }
                    }, {
                      "expression" : "recordProjection",
                      "e" : {
                        "expression" : "argument",
                        "attributes" : [ {
                          "attrName" : "profit",
                          "relName" : "subsetPelagoUnpack#5484"
                        } ],
                        "type" : {
                          "relName" : "subsetPelagoUnpack#5484",
                          "type" : "record"
                        },
                        "argNo" : -1
                      },
                      "attribute" : {
                        "attrName" : "profit",
                        "relName" : "subsetPelagoUnpack#5484"
                      }
                    } ],
                    "input" : {
                      "operator" : "cpu-to-gpu",
                      "projections" : [ {
                        "relName" : "subsetPelagoUnpack#5484",
                        "attrName" : "d_year",
                        "isBlock" : true
                      }, {
                        "relName" : "subsetPelagoUnpack#5484",
                        "attrName" : "s_city",
                        "isBlock" : true
                      }, {
                        "relName" : "subsetPelagoUnpack#5484",
                        "attrName" : "p_brand1",
                        "isBlock" : true
                      }, {
                        "relName" : "subsetPelagoUnpack#5484",
                        "attrName" : "profit",
                        "isBlock" : true
                      } ],
                      "queueSize" : 262144,
                      "granularity" : "thread",
                      "input" : {
                        "operator" : "mem-move-device",
                        "projections" : [ {
                          "relName" : "subsetPelagoUnpack#5484",
                          "attrName" : "d_year",
                          "isBlock" : true
                        }, {
                          "relName" : "subsetPelagoUnpack#5484",
                          "attrName" : "s_city",
                          "isBlock" : true
                        }, {
                          "relName" : "subsetPelagoUnpack#5484",
                          "attrName" : "p_brand1",
                          "isBlock" : true
                        }, {
                          "relName" : "subsetPelagoUnpack#5484",
                          "attrName" : "profit",
                          "isBlock" : true
                        } ],
                        "input" : {
                          "operator" : "router",
                          "gpu" : false,
                          "projections" : [ {
                            "relName" : "subsetPelagoUnpack#5484",
                            "attrName" : "d_year",
                            "isBlock" : true
                          }, {
                            "relName" : "subsetPelagoUnpack#5484",
                            "attrName" : "s_city",
                            "isBlock" : true
                          }, {
                            "relName" : "subsetPelagoUnpack#5484",
                            "attrName" : "p_brand1",
                            "isBlock" : true
                          }, {
                            "relName" : "subsetPelagoUnpack#5484",
                            "attrName" : "profit",
                            "isBlock" : true
                          } ],
                          "numOfParents" : 1,
                          "producers" : 2,
                          "slack" : 8,
                          "cpu_targets" : false,
                          "input" : {
                            "operator" : "mem-move-device",
                            "projections" : [ {
                              "relName" : "subsetPelagoUnpack#5484",
                              "attrName" : "d_year",
                              "isBlock" : true
                            }, {
                              "relName" : "subsetPelagoUnpack#5484",
                              "attrName" : "s_city",
                              "isBlock" : true
                            }, {
                              "relName" : "subsetPelagoUnpack#5484",
                              "attrName" : "p_brand1",
                              "isBlock" : true
                            }, {
                              "relName" : "subsetPelagoUnpack#5484",
                              "attrName" : "profit",
                              "isBlock" : true
                            } ],
                            "input" : {
                              "operator" : "gpu-to-cpu",
                              "projections" : [ {
                                "relName" : "subsetPelagoUnpack#5484",
                                "attrName" : "d_year",
                                "isBlock" : true
                              }, {
                                "relName" : "subsetPelagoUnpack#5484",
                                "attrName" : "s_city",
                                "isBlock" : true
                              }, {
                                "relName" : "subsetPelagoUnpack#5484",
                                "attrName" : "p_brand1",
                                "isBlock" : true
                              }, {
                                "relName" : "subsetPelagoUnpack#5484",
                                "attrName" : "profit",
                                "isBlock" : true
                              } ],
                              "queueSize" : 262144,
                              "granularity" : "thread",
                              "input" : {
                                "operator" : "pack",
                                "gpu" : true,
                                "projections" : [ {
                                  "expression" : "recordProjection",
                                  "e" : {
                                    "expression" : "argument",
                                    "attributes" : [ {
                                      "attrName" : "d_year",
                                      "relName" : "subsetPelagoUnpack#5484"
                                    } ],
                                    "type" : {
                                      "relName" : "subsetPelagoUnpack#5484",
                                      "type" : "record"
                                    },
                                    "argNo" : -1
                                  },
                                  "attribute" : {
                                    "attrName" : "d_year",
                                    "relName" : "subsetPelagoUnpack#5484"
                                  }
                                }, {
                                  "expression" : "recordProjection",
                                  "e" : {
                                    "expression" : "argument",
                                    "attributes" : [ {
                                      "attrName" : "s_city",
                                      "relName" : "subsetPelagoUnpack#5484"
                                    } ],
                                    "type" : {
                                      "relName" : "subsetPelagoUnpack#5484",
                                      "type" : "record"
                                    },
                                    "argNo" : -1
                                  },
                                  "attribute" : {
                                    "attrName" : "s_city",
                                    "relName" : "subsetPelagoUnpack#5484"
                                  }
                                }, {
                                  "expression" : "recordProjection",
                                  "e" : {
                                    "expression" : "argument",
                                    "attributes" : [ {
                                      "attrName" : "p_brand1",
                                      "relName" : "subsetPelagoUnpack#5484"
                                    } ],
                                    "type" : {
                                      "relName" : "subsetPelagoUnpack#5484",
                                      "type" : "record"
                                    },
                                    "argNo" : -1
                                  },
                                  "attribute" : {
                                    "attrName" : "p_brand1",
                                    "relName" : "subsetPelagoUnpack#5484"
                                  }
                                }, {
                                  "expression" : "recordProjection",
                                  "e" : {
                                    "expression" : "argument",
                                    "attributes" : [ {
                                      "attrName" : "profit",
                                      "relName" : "subsetPelagoUnpack#5484"
                                    } ],
                                    "type" : {
                                      "relName" : "subsetPelagoUnpack#5484",
                                      "type" : "record"
                                    },
                                    "argNo" : -1
                                  },
                                  "attribute" : {
                                    "attrName" : "profit",
                                    "relName" : "subsetPelagoUnpack#5484"
                                  }
                                } ],
                                "input" : {
                                  "operator" : "groupby",
                                  "gpu" : true,
                                  "k" : [ {
                                    "expression" : "recordProjection",
                                    "e" : {
                                      "expression" : "argument",
                                      "attributes" : [ {
                                        "attrName" : "d_year",
                                        "relName" : "subsetPelagoProject#5478"
                                      } ],
                                      "type" : {
                                        "relName" : "subsetPelagoProject#5478",
                                        "type" : "record"
                                      },
                                      "argNo" : -1
                                    },
                                    "attribute" : {
                                      "attrName" : "d_year",
                                      "relName" : "subsetPelagoProject#5478"
                                    },
                                    "register_as" : {
                                      "attrName" : "d_year",
                                      "relName" : "subsetPelagoUnpack#5484"
                                    }
                                  }, {
                                    "expression" : "recordProjection",
                                    "e" : {
                                      "expression" : "argument",
                                      "attributes" : [ {
                                        "attrName" : "s_city",
                                        "relName" : "subsetPelagoProject#5478"
                                      } ],
                                      "type" : {
                                        "relName" : "subsetPelagoProject#5478",
                                        "type" : "record"
                                      },
                                      "argNo" : -1
                                    },
                                    "attribute" : {
                                      "attrName" : "s_city",
                                      "relName" : "subsetPelagoProject#5478"
                                    },
                                    "register_as" : {
                                      "attrName" : "s_city",
                                      "relName" : "subsetPelagoUnpack#5484"
                                    }
                                  }, {
                                    "expression" : "recordProjection",
                                    "e" : {
                                      "expression" : "argument",
                                      "attributes" : [ {
                                        "attrName" : "p_brand1",
                                        "relName" : "subsetPelagoProject#5478"
                                      } ],
                                      "type" : {
                                        "relName" : "subsetPelagoProject#5478",
                                        "type" : "record"
                                      },
                                      "argNo" : -1
                                    },
                                    "attribute" : {
                                      "attrName" : "p_brand1",
                                      "relName" : "subsetPelagoProject#5478"
                                    },
                                    "register_as" : {
                                      "attrName" : "p_brand1",
                                      "relName" : "subsetPelagoUnpack#5484"
                                    }
                                  } ],
                                  "e" : [ {
                                    "m" : "sum",
                                    "e" : {
                                      "expression" : "recordProjection",
                                      "e" : {
                                        "expression" : "argument",
                                        "attributes" : [ {
                                          "attrName" : "$f3",
                                          "relName" : "subsetPelagoProject#5478"
                                        } ],
                                        "type" : {
                                          "relName" : "subsetPelagoProject#5478",
                                          "type" : "record"
                                        },
                                        "argNo" : -1
                                      },
                                      "attribute" : {
                                        "attrName" : "$f3",
                                        "relName" : "subsetPelagoProject#5478"
                                      },
                                      "register_as" : {
                                        "attrName" : "profit",
                                        "relName" : "subsetPelagoUnpack#5484"
                                      }
                                    },
                                    "packet" : 1,
                                    "offset" : 0
                                  } ],
                                  "hash_bits" : 10,
                                  "maxInputSize" : 131072,
                                  "input" : {
                                    "operator" : "project",
                                    "gpu" : true,
                                    "relName" : "subsetPelagoProject#5478",
                                    "e" : [ {
                                      "expression" : "recordProjection",
                                      "e" : {
                                        "expression" : "argument",
                                        "attributes" : [ {
                                          "attrName" : "d_year",
                                          "relName" : "subsetPelagoProject#5478"
                                        } ],
                                        "type" : {
                                          "relName" : "subsetPelagoProject#5478",
                                          "type" : "record"
                                        },
                                        "argNo" : -1
                                      },
                                      "attribute" : {
                                        "attrName" : "d_year",
                                        "relName" : "subsetPelagoProject#5478"
                                      },
                                      "register_as" : {
                                        "attrName" : "d_year",
                                        "relName" : "subsetPelagoProject#5478"
                                      }
                                    }, {
                                      "expression" : "recordProjection",
                                      "e" : {
                                        "expression" : "argument",
                                        "attributes" : [ {
                                          "attrName" : "s_city",
                                          "relName" : "subsetPelagoProject#5478"
                                        } ],
                                        "type" : {
                                          "relName" : "subsetPelagoProject#5478",
                                          "type" : "record"
                                        },
                                        "argNo" : -1
                                      },
                                      "attribute" : {
                                        "attrName" : "s_city",
                                        "relName" : "subsetPelagoProject#5478"
                                      },
                                      "register_as" : {
                                        "attrName" : "s_city",
                                        "relName" : "subsetPelagoProject#5478"
                                      }
                                    }, {
                                      "expression" : "recordProjection",
                                      "e" : {
                                        "expression" : "argument",
                                        "attributes" : [ {
                                          "attrName" : "p_brand1",
                                          "relName" : "subsetPelagoProject#5478"
                                        } ],
                                        "type" : {
                                          "relName" : "subsetPelagoProject#5478",
                                          "type" : "record"
                                        },
                                        "argNo" : -1
                                      },
                                      "attribute" : {
                                        "attrName" : "p_brand1",
                                        "relName" : "subsetPelagoProject#5478"
                                      },
                                      "register_as" : {
                                        "attrName" : "p_brand1",
                                        "relName" : "subsetPelagoProject#5478"
                                      }
                                    }, {
                                      "expression" : "recordProjection",
                                      "e" : {
                                        "expression" : "argument",
                                        "attributes" : [ {
                                          "attrName" : "-",
                                          "relName" : "subsetPelagoProject#5478"
                                        } ],
                                        "type" : {
                                          "relName" : "subsetPelagoProject#5478",
                                          "type" : "record"
                                        },
                                        "argNo" : -1
                                      },
                                      "attribute" : {
                                        "attrName" : "-",
                                        "relName" : "subsetPelagoProject#5478"
                                      },
                                      "register_as" : {
                                        "attrName" : "$f3",
                                        "relName" : "subsetPelagoProject#5478"
                                      }
                                    } ],
                                    "input" : {
                                      "operator" : "hashjoin-chained",
                                      "gpu" : true,
                                      "build_k" : {
                                        "expression" : "recordProjection",
                                        "e" : {
                                          "expression" : "argument",
                                          "attributes" : [ {
                                            "attrName" : "c_custkey",
                                            "relName" : "subsetPelagoProject#5452"
                                          } ],
                                          "type" : {
                                            "relName" : "subsetPelagoProject#5452",
                                            "type" : "record"
                                          },
                                          "argNo" : -1
                                        },
                                        "attribute" : {
                                          "attrName" : "c_custkey",
                                          "relName" : "subsetPelagoProject#5452"
                                        },
                                        "register_as" : {
                                          "attrName" : "$1",
                                          "relName" : "subsetPelagoProject#5478"
                                        }
                                      },
                                      "build_e" : [ {
                                        "e" : {
                                          "expression" : "recordProjection",
                                          "e" : {
                                            "expression" : "argument",
                                            "attributes" : [ {
                                              "attrName" : "c_custkey",
                                              "relName" : "subsetPelagoProject#5452"
                                            } ],
                                            "type" : {
                                              "relName" : "subsetPelagoProject#5452",
                                              "type" : "record"
                                            },
                                            "argNo" : -1
                                          },
                                          "attribute" : {
                                            "attrName" : "c_custkey",
                                            "relName" : "subsetPelagoProject#5452"
                                          },
                                          "register_as" : {
                                            "attrName" : "c_custkey",
                                            "relName" : "subsetPelagoProject#5478"
                                          }
                                        },
                                        "packet" : 1,
                                        "offset" : 0
                                      } ],
                                      "build_w" : [ 64, 32 ],
                                      "build_input" : {
                                        "operator" : "project",
                                        "gpu" : true,
                                        "relName" : "subsetPelagoProject#5452",
                                        "e" : [ {
                                          "expression" : "recordProjection",
                                          "e" : {
                                            "expression" : "argument",
                                            "attributes" : [ {
                                              "attrName" : "c_custkey",
                                              "relName" : "inputs/ssbm1000/customer.csv"
                                            } ],
                                            "type" : {
                                              "relName" : "inputs/ssbm1000/customer.csv",
                                              "type" : "record"
                                            },
                                            "argNo" : -1
                                          },
                                          "attribute" : {
                                            "attrName" : "c_custkey",
                                            "relName" : "inputs/ssbm1000/customer.csv"
                                          },
                                          "register_as" : {
                                            "attrName" : "c_custkey",
                                            "relName" : "subsetPelagoProject#5452"
                                          }
                                        } ],
                                        "input" : {
                                          "operator" : "select",
                                          "gpu" : true,
                                          "p" : {
                                            "expression" : "eq",
                                            "left" : {
                                              "expression" : "recordProjection",
                                              "e" : {
                                                "expression" : "argument",
                                                "attributes" : [ {
                                                  "attrName" : "c_region",
                                                  "relName" : "inputs/ssbm1000/customer.csv"
                                                } ],
                                                "type" : {
                                                  "relName" : "inputs/ssbm1000/customer.csv",
                                                  "type" : "record"
                                                },
                                                "argNo" : -1
                                              },
                                              "attribute" : {
                                                "attrName" : "c_region",
                                                "relName" : "inputs/ssbm1000/customer.csv"
                                              }
                                            },
                                            "right" : {
                                              "expression" : "dstring",
                                              "v" : "AMERICA",
                                              "dict" : {
                                                "path" : "inputs/ssbm1000/customer.csv.ProjectedRelDataTypeField(#1: c_region VARCHAR,null).dict"
                                              }
                                            }
                                          },
                                          "input" : {
                                            "operator" : "unpack",
                                            "gpu" : true,
                                            "projections" : [ {
                                              "expression" : "recordProjection",
                                              "e" : {
                                                "expression" : "argument",
                                                "attributes" : [ {
                                                  "attrName" : "c_custkey",
                                                  "relName" : "inputs/ssbm1000/customer.csv"
                                                } ],
                                                "type" : {
                                                  "relName" : "inputs/ssbm1000/customer.csv",
                                                  "type" : "record"
                                                },
                                                "argNo" : -1
                                              },
                                              "attribute" : {
                                                "attrName" : "c_custkey",
                                                "relName" : "inputs/ssbm1000/customer.csv"
                                              }
                                            }, {
                                              "expression" : "recordProjection",
                                              "e" : {
                                                "expression" : "argument",
                                                "attributes" : [ {
                                                  "attrName" : "c_region",
                                                  "relName" : "inputs/ssbm1000/customer.csv"
                                                } ],
                                                "type" : {
                                                  "relName" : "inputs/ssbm1000/customer.csv",
                                                  "type" : "record"
                                                },
                                                "argNo" : -1
                                              },
                                              "attribute" : {
                                                "attrName" : "c_region",
                                                "relName" : "inputs/ssbm1000/customer.csv"
                                              }
                                            } ],
                                            "input" : {
                                              "operator" : "cpu-to-gpu",
                                              "projections" : [ {
                                                "relName" : "inputs/ssbm1000/customer.csv",
                                                "attrName" : "c_custkey",
                                                "isBlock" : true
                                              }, {
                                                "relName" : "inputs/ssbm1000/customer.csv",
                                                "attrName" : "c_region",
                                                "isBlock" : true
                                              } ],
                                              "queueSize" : 262144,
                                              "granularity" : "thread",
                                              "input" : {
                                                "operator" : "router",
                                                "gpu" : false,
                                                "projections" : [ {
                                                  "relName" : "inputs/ssbm1000/customer.csv",
                                                  "attrName" : "c_custkey",
                                                  "isBlock" : true
                                                }, {
                                                  "relName" : "inputs/ssbm1000/customer.csv",
                                                  "attrName" : "c_region",
                                                  "isBlock" : true
                                                } ],
                                                "numOfParents" : 2,
                                                "producers" : 1,
                                                "slack" : 8,
                                                "cpu_targets" : false,
                                                "target" : {
                                                  "expression" : "recordProjection",
                                                  "e" : {
                                                    "expression" : "argument",
                                                    "argNo" : -1,
                                                    "type" : {
                                                      "type" : "record",
                                                      "relName" : "inputs/ssbm1000/customer.csv"
                                                    },
                                                    "attributes" : [ {
                                                      "relName" : "inputs/ssbm1000/customer.csv",
                                                      "attrName" : "__broadcastTarget"
                                                    } ]
                                                  },
                                                  "attribute" : {
                                                    "relName" : "inputs/ssbm1000/customer.csv",
                                                    "attrName" : "__broadcastTarget"
                                                  }
                                                },
                                                "input" : {
                                                  "operator" : "mem-broadcast-device",
                                                  "num_of_targets" : 2,
                                                  "projections" : [ {
                                                    "relName" : "inputs/ssbm1000/customer.csv",
                                                    "attrName" : "c_custkey",
                                                    "isBlock" : true
                                                  }, {
                                                    "relName" : "inputs/ssbm1000/customer.csv",
                                                    "attrName" : "c_region",
                                                    "isBlock" : true
                                                  } ],
                                                  "input" : {
                                                    "operator" : "scan",
                                                    "gpu" : false,
                                                    "plugin" : {
                                                      "type" : "block",
                                                      "linehint" : 30000000,
                                                      "name" : "inputs/ssbm1000/customer.csv",
                                                      "projections" : [ {
                                                        "relName" : "inputs/ssbm1000/customer.csv",
                                                        "attrName" : "c_custkey"
                                                      }, {
                                                        "relName" : "inputs/ssbm1000/customer.csv",
                                                        "attrName" : "c_region"
                                                      } ],
                                                      "schema" : [ {
                                                        "relName" : "inputs/ssbm1000/customer.csv",
                                                        "attrName" : "c_custkey",
                                                        "type" : {
                                                          "type" : "int"
                                                        },
                                                        "attrNo" : 1
                                                      }, {
                                                        "relName" : "inputs/ssbm1000/customer.csv",
                                                        "attrName" : "c_name",
                                                        "type" : {
                                                          "type" : "dstring"
                                                        },
                                                        "attrNo" : 2
                                                      }, {
                                                        "relName" : "inputs/ssbm1000/customer.csv",
                                                        "attrName" : "c_address",
                                                        "type" : {
                                                          "type" : "dstring"
                                                        },
                                                        "attrNo" : 3
                                                      }, {
                                                        "relName" : "inputs/ssbm1000/customer.csv",
                                                        "attrName" : "c_city",
                                                        "type" : {
                                                          "type" : "dstring"
                                                        },
                                                        "attrNo" : 4
                                                      }, {
                                                        "relName" : "inputs/ssbm1000/customer.csv",
                                                        "attrName" : "c_nation",
                                                        "type" : {
                                                          "type" : "dstring"
                                                        },
                                                        "attrNo" : 5
                                                      }, {
                                                        "relName" : "inputs/ssbm1000/customer.csv",
                                                        "attrName" : "c_region",
                                                        "type" : {
                                                          "type" : "dstring"
                                                        },
                                                        "attrNo" : 6
                                                      }, {
                                                        "relName" : "inputs/ssbm1000/customer.csv",
                                                        "attrName" : "c_phone",
                                                        "type" : {
                                                          "type" : "dstring"
                                                        },
                                                        "attrNo" : 7
                                                      }, {
                                                        "relName" : "inputs/ssbm1000/customer.csv",
                                                        "attrName" : "c_mktsegment",
                                                        "type" : {
                                                          "type" : "dstring"
                                                        },
                                                        "attrNo" : 8
                                                      } ]
                                                    }
                                                  },
                                                  "to_cpu" : false
                                                }
                                              }
                                            }
                                          }
                                        }
                                      },
                                      "probe_k" : {
                                        "expression" : "recordProjection",
                                        "e" : {
                                          "expression" : "argument",
                                          "attributes" : [ {
                                            "attrName" : "lo_custkey",
                                            "relName" : "subsetPelagoProject#5476"
                                          } ],
                                          "type" : {
                                            "relName" : "subsetPelagoProject#5476",
                                            "type" : "record"
                                          },
                                          "argNo" : -1
                                        },
                                        "attribute" : {
                                          "attrName" : "lo_custkey",
                                          "relName" : "subsetPelagoProject#5476"
                                        },
                                        "register_as" : {
                                          "attrName" : "$0",
                                          "relName" : "subsetPelagoProject#5478"
                                        }
                                      },
                                      "probe_e" : [ {
                                        "e" : {
                                          "expression" : "recordProjection",
                                          "e" : {
                                            "expression" : "argument",
                                            "attributes" : [ {
                                              "attrName" : "lo_custkey",
                                              "relName" : "subsetPelagoProject#5476"
                                            } ],
                                            "type" : {
                                              "relName" : "subsetPelagoProject#5476",
                                              "type" : "record"
                                            },
                                            "argNo" : -1
                                          },
                                          "attribute" : {
                                            "attrName" : "lo_custkey",
                                            "relName" : "subsetPelagoProject#5476"
                                          },
                                          "register_as" : {
                                            "attrName" : "lo_custkey",
                                            "relName" : "subsetPelagoProject#5478"
                                          }
                                        },
                                        "packet" : 1,
                                        "offset" : 0
                                      }, {
                                        "e" : {
                                          "expression" : "recordProjection",
                                          "e" : {
                                            "expression" : "argument",
                                            "attributes" : [ {
                                              "attrName" : "d_year",
                                              "relName" : "subsetPelagoProject#5476"
                                            } ],
                                            "type" : {
                                              "relName" : "subsetPelagoProject#5476",
                                              "type" : "record"
                                            },
                                            "argNo" : -1
                                          },
                                          "attribute" : {
                                            "attrName" : "d_year",
                                            "relName" : "subsetPelagoProject#5476"
                                          },
                                          "register_as" : {
                                            "attrName" : "d_year",
                                            "relName" : "subsetPelagoProject#5478"
                                          }
                                        },
                                        "packet" : 2,
                                        "offset" : 0
                                      }, {
                                        "e" : {
                                          "expression" : "recordProjection",
                                          "e" : {
                                            "expression" : "argument",
                                            "attributes" : [ {
                                              "attrName" : "s_city",
                                              "relName" : "subsetPelagoProject#5476"
                                            } ],
                                            "type" : {
                                              "relName" : "subsetPelagoProject#5476",
                                              "type" : "record"
                                            },
                                            "argNo" : -1
                                          },
                                          "attribute" : {
                                            "attrName" : "s_city",
                                            "relName" : "subsetPelagoProject#5476"
                                          },
                                          "register_as" : {
                                            "attrName" : "s_city",
                                            "relName" : "subsetPelagoProject#5478"
                                          }
                                        },
                                        "packet" : 3,
                                        "offset" : 0
                                      }, {
                                        "e" : {
                                          "expression" : "recordProjection",
                                          "e" : {
                                            "expression" : "argument",
                                            "attributes" : [ {
                                              "attrName" : "p_brand1",
                                              "relName" : "subsetPelagoProject#5476"
                                            } ],
                                            "type" : {
                                              "relName" : "subsetPelagoProject#5476",
                                              "type" : "record"
                                            },
                                            "argNo" : -1
                                          },
                                          "attribute" : {
                                            "attrName" : "p_brand1",
                                            "relName" : "subsetPelagoProject#5476"
                                          },
                                          "register_as" : {
                                            "attrName" : "p_brand1",
                                            "relName" : "subsetPelagoProject#5478"
                                          }
                                        },
                                        "packet" : 4,
                                        "offset" : 0
                                      }, {
                                        "e" : {
                                          "expression" : "recordProjection",
                                          "e" : {
                                            "expression" : "argument",
                                            "attributes" : [ {
                                              "attrName" : "-",
                                              "relName" : "subsetPelagoProject#5476"
                                            } ],
                                            "type" : {
                                              "relName" : "subsetPelagoProject#5476",
                                              "type" : "record"
                                            },
                                            "argNo" : -1
                                          },
                                          "attribute" : {
                                            "attrName" : "-",
                                            "relName" : "subsetPelagoProject#5476"
                                          },
                                          "register_as" : {
                                            "attrName" : "-",
                                            "relName" : "subsetPelagoProject#5478"
                                          }
                                        },
                                        "packet" : 5,
                                        "offset" : 0
                                      } ],
                                      "probe_w" : [ 64, 32, 32, 32, 32, 32 ],
                                      "hash_bits" : 12,
                                      "maxBuildInputSize" : 30000000,
                                      "probe_input" : {
                                        "operator" : "project",
                                        "gpu" : true,
                                        "relName" : "subsetPelagoProject#5476",
                                        "e" : [ {
                                          "expression" : "recordProjection",
                                          "e" : {
                                            "expression" : "argument",
                                            "attributes" : [ {
                                              "attrName" : "lo_custkey",
                                              "relName" : "subsetPelagoProject#5476"
                                            } ],
                                            "type" : {
                                              "relName" : "subsetPelagoProject#5476",
                                              "type" : "record"
                                            },
                                            "argNo" : -1
                                          },
                                          "attribute" : {
                                            "attrName" : "lo_custkey",
                                            "relName" : "subsetPelagoProject#5476"
                                          },
                                          "register_as" : {
                                            "attrName" : "lo_custkey",
                                            "relName" : "subsetPelagoProject#5476"
                                          }
                                        }, {
                                          "expression" : "recordProjection",
                                          "e" : {
                                            "expression" : "argument",
                                            "attributes" : [ {
                                              "attrName" : "d_year",
                                              "relName" : "subsetPelagoProject#5476"
                                            } ],
                                            "type" : {
                                              "relName" : "subsetPelagoProject#5476",
                                              "type" : "record"
                                            },
                                            "argNo" : -1
                                          },
                                          "attribute" : {
                                            "attrName" : "d_year",
                                            "relName" : "subsetPelagoProject#5476"
                                          },
                                          "register_as" : {
                                            "attrName" : "d_year",
                                            "relName" : "subsetPelagoProject#5476"
                                          }
                                        }, {
                                          "expression" : "recordProjection",
                                          "e" : {
                                            "expression" : "argument",
                                            "attributes" : [ {
                                              "attrName" : "s_city",
                                              "relName" : "subsetPelagoProject#5476"
                                            } ],
                                            "type" : {
                                              "relName" : "subsetPelagoProject#5476",
                                              "type" : "record"
                                            },
                                            "argNo" : -1
                                          },
                                          "attribute" : {
                                            "attrName" : "s_city",
                                            "relName" : "subsetPelagoProject#5476"
                                          },
                                          "register_as" : {
                                            "attrName" : "s_city",
                                            "relName" : "subsetPelagoProject#5476"
                                          }
                                        }, {
                                          "expression" : "recordProjection",
                                          "e" : {
                                            "expression" : "argument",
                                            "attributes" : [ {
                                              "attrName" : "p_brand1",
                                              "relName" : "subsetPelagoProject#5476"
                                            } ],
                                            "type" : {
                                              "relName" : "subsetPelagoProject#5476",
                                              "type" : "record"
                                            },
                                            "argNo" : -1
                                          },
                                          "attribute" : {
                                            "attrName" : "p_brand1",
                                            "relName" : "subsetPelagoProject#5476"
                                          },
                                          "register_as" : {
                                            "attrName" : "p_brand1",
                                            "relName" : "subsetPelagoProject#5476"
                                          }
                                        }, {
                                          "expression" : "recordProjection",
                                          "e" : {
                                            "expression" : "argument",
                                            "attributes" : [ {
                                              "attrName" : "-",
                                              "relName" : "subsetPelagoProject#5476"
                                            } ],
                                            "type" : {
                                              "relName" : "subsetPelagoProject#5476",
                                              "type" : "record"
                                            },
                                            "argNo" : -1
                                          },
                                          "attribute" : {
                                            "attrName" : "-",
                                            "relName" : "subsetPelagoProject#5476"
                                          },
                                          "register_as" : {
                                            "attrName" : "-",
                                            "relName" : "subsetPelagoProject#5476"
                                          }
                                        } ],
                                        "input" : {
                                          "operator" : "hashjoin-chained",
                                          "gpu" : true,
                                          "build_k" : {
                                            "expression" : "recordProjection",
                                            "e" : {
                                              "expression" : "argument",
                                              "attributes" : [ {
                                                "attrName" : "p_partkey",
                                                "relName" : "subsetPelagoProject#5457"
                                              } ],
                                              "type" : {
                                                "relName" : "subsetPelagoProject#5457",
                                                "type" : "record"
                                              },
                                              "argNo" : -1
                                            },
                                            "attribute" : {
                                              "attrName" : "p_partkey",
                                              "relName" : "subsetPelagoProject#5457"
                                            },
                                            "register_as" : {
                                              "attrName" : "$3",
                                              "relName" : "subsetPelagoProject#5476"
                                            }
                                          },
                                          "build_e" : [ {
                                            "e" : {
                                              "expression" : "recordProjection",
                                              "e" : {
                                                "expression" : "argument",
                                                "attributes" : [ {
                                                  "attrName" : "p_partkey",
                                                  "relName" : "subsetPelagoProject#5457"
                                                } ],
                                                "type" : {
                                                  "relName" : "subsetPelagoProject#5457",
                                                  "type" : "record"
                                                },
                                                "argNo" : -1
                                              },
                                              "attribute" : {
                                                "attrName" : "p_partkey",
                                                "relName" : "subsetPelagoProject#5457"
                                              },
                                              "register_as" : {
                                                "attrName" : "p_partkey",
                                                "relName" : "subsetPelagoProject#5476"
                                              }
                                            },
                                            "packet" : 1,
                                            "offset" : 0
                                          }, {
                                            "e" : {
                                              "expression" : "recordProjection",
                                              "e" : {
                                                "expression" : "argument",
                                                "attributes" : [ {
                                                  "attrName" : "p_brand1",
                                                  "relName" : "subsetPelagoProject#5457"
                                                } ],
                                                "type" : {
                                                  "relName" : "subsetPelagoProject#5457",
                                                  "type" : "record"
                                                },
                                                "argNo" : -1
                                              },
                                              "attribute" : {
                                                "attrName" : "p_brand1",
                                                "relName" : "subsetPelagoProject#5457"
                                              },
                                              "register_as" : {
                                                "attrName" : "p_brand1",
                                                "relName" : "subsetPelagoProject#5476"
                                              }
                                            },
                                            "packet" : 2,
                                            "offset" : 0
                                          } ],
                                          "build_w" : [ 64, 32, 32 ],
                                          "build_input" : {
                                            "operator" : "project",
                                            "gpu" : true,
                                            "relName" : "subsetPelagoProject#5457",
                                            "e" : [ {
                                              "expression" : "recordProjection",
                                              "e" : {
                                                "expression" : "argument",
                                                "attributes" : [ {
                                                  "attrName" : "p_partkey",
                                                  "relName" : "inputs/ssbm1000/part.csv"
                                                } ],
                                                "type" : {
                                                  "relName" : "inputs/ssbm1000/part.csv",
                                                  "type" : "record"
                                                },
                                                "argNo" : -1
                                              },
                                              "attribute" : {
                                                "attrName" : "p_partkey",
                                                "relName" : "inputs/ssbm1000/part.csv"
                                              },
                                              "register_as" : {
                                                "attrName" : "p_partkey",
                                                "relName" : "subsetPelagoProject#5457"
                                              }
                                            }, {
                                              "expression" : "recordProjection",
                                              "e" : {
                                                "expression" : "argument",
                                                "attributes" : [ {
                                                  "attrName" : "p_brand1",
                                                  "relName" : "inputs/ssbm1000/part.csv"
                                                } ],
                                                "type" : {
                                                  "relName" : "inputs/ssbm1000/part.csv",
                                                  "type" : "record"
                                                },
                                                "argNo" : -1
                                              },
                                              "attribute" : {
                                                "attrName" : "p_brand1",
                                                "relName" : "inputs/ssbm1000/part.csv"
                                              },
                                              "register_as" : {
                                                "attrName" : "p_brand1",
                                                "relName" : "subsetPelagoProject#5457"
                                              }
                                            } ],
                                            "input" : {
                                              "operator" : "select",
                                              "gpu" : true,
                                              "p" : {
                                                "expression" : "eq",
                                                "left" : {
                                                  "expression" : "recordProjection",
                                                  "e" : {
                                                    "expression" : "argument",
                                                    "attributes" : [ {
                                                      "attrName" : "p_category",
                                                      "relName" : "inputs/ssbm1000/part.csv"
                                                    } ],
                                                    "type" : {
                                                      "relName" : "inputs/ssbm1000/part.csv",
                                                      "type" : "record"
                                                    },
                                                    "argNo" : -1
                                                  },
                                                  "attribute" : {
                                                    "attrName" : "p_category",
                                                    "relName" : "inputs/ssbm1000/part.csv"
                                                  }
                                                },
                                                "right" : {
                                                  "expression" : "dstring",
                                                  "v" : "MFGR#14",
                                                  "dict" : {
                                                    "path" : "inputs/ssbm1000/part.csv.ProjectedRelDataTypeField(#1: p_category VARCHAR,null).dict"
                                                  }
                                                }
                                              },
                                              "input" : {
                                                "operator" : "unpack",
                                                "gpu" : true,
                                                "projections" : [ {
                                                  "expression" : "recordProjection",
                                                  "e" : {
                                                    "expression" : "argument",
                                                    "attributes" : [ {
                                                      "attrName" : "p_partkey",
                                                      "relName" : "inputs/ssbm1000/part.csv"
                                                    } ],
                                                    "type" : {
                                                      "relName" : "inputs/ssbm1000/part.csv",
                                                      "type" : "record"
                                                    },
                                                    "argNo" : -1
                                                  },
                                                  "attribute" : {
                                                    "attrName" : "p_partkey",
                                                    "relName" : "inputs/ssbm1000/part.csv"
                                                  }
                                                }, {
                                                  "expression" : "recordProjection",
                                                  "e" : {
                                                    "expression" : "argument",
                                                    "attributes" : [ {
                                                      "attrName" : "p_category",
                                                      "relName" : "inputs/ssbm1000/part.csv"
                                                    } ],
                                                    "type" : {
                                                      "relName" : "inputs/ssbm1000/part.csv",
                                                      "type" : "record"
                                                    },
                                                    "argNo" : -1
                                                  },
                                                  "attribute" : {
                                                    "attrName" : "p_category",
                                                    "relName" : "inputs/ssbm1000/part.csv"
                                                  }
                                                }, {
                                                  "expression" : "recordProjection",
                                                  "e" : {
                                                    "expression" : "argument",
                                                    "attributes" : [ {
                                                      "attrName" : "p_brand1",
                                                      "relName" : "inputs/ssbm1000/part.csv"
                                                    } ],
                                                    "type" : {
                                                      "relName" : "inputs/ssbm1000/part.csv",
                                                      "type" : "record"
                                                    },
                                                    "argNo" : -1
                                                  },
                                                  "attribute" : {
                                                    "attrName" : "p_brand1",
                                                    "relName" : "inputs/ssbm1000/part.csv"
                                                  }
                                                } ],
                                                "input" : {
                                                  "operator" : "cpu-to-gpu",
                                                  "projections" : [ {
                                                    "relName" : "inputs/ssbm1000/part.csv",
                                                    "attrName" : "p_partkey",
                                                    "isBlock" : true
                                                  }, {
                                                    "relName" : "inputs/ssbm1000/part.csv",
                                                    "attrName" : "p_category",
                                                    "isBlock" : true
                                                  }, {
                                                    "relName" : "inputs/ssbm1000/part.csv",
                                                    "attrName" : "p_brand1",
                                                    "isBlock" : true
                                                  } ],
                                                  "queueSize" : 262144,
                                                  "granularity" : "thread",
                                                  "input" : {
                                                    "operator" : "router",
                                                    "gpu" : false,
                                                    "projections" : [ {
                                                      "relName" : "inputs/ssbm1000/part.csv",
                                                      "attrName" : "p_partkey",
                                                      "isBlock" : true
                                                    }, {
                                                      "relName" : "inputs/ssbm1000/part.csv",
                                                      "attrName" : "p_category",
                                                      "isBlock" : true
                                                    }, {
                                                      "relName" : "inputs/ssbm1000/part.csv",
                                                      "attrName" : "p_brand1",
                                                      "isBlock" : true
                                                    } ],
                                                    "numOfParents" : 2,
                                                    "producers" : 1,
                                                    "slack" : 8,
                                                    "cpu_targets" : false,
                                                    "target" : {
                                                      "expression" : "recordProjection",
                                                      "e" : {
                                                        "expression" : "argument",
                                                        "argNo" : -1,
                                                        "type" : {
                                                          "type" : "record",
                                                          "relName" : "inputs/ssbm1000/part.csv"
                                                        },
                                                        "attributes" : [ {
                                                          "relName" : "inputs/ssbm1000/part.csv",
                                                          "attrName" : "__broadcastTarget"
                                                        } ]
                                                      },
                                                      "attribute" : {
                                                        "relName" : "inputs/ssbm1000/part.csv",
                                                        "attrName" : "__broadcastTarget"
                                                      }
                                                    },
                                                    "input" : {
                                                      "operator" : "mem-broadcast-device",
                                                      "num_of_targets" : 2,
                                                      "projections" : [ {
                                                        "relName" : "inputs/ssbm1000/part.csv",
                                                        "attrName" : "p_partkey",
                                                        "isBlock" : true
                                                      }, {
                                                        "relName" : "inputs/ssbm1000/part.csv",
                                                        "attrName" : "p_category",
                                                        "isBlock" : true
                                                      }, {
                                                        "relName" : "inputs/ssbm1000/part.csv",
                                                        "attrName" : "p_brand1",
                                                        "isBlock" : true
                                                      } ],
                                                      "input" : {
                                                        "operator" : "scan",
                                                        "gpu" : false,
                                                        "plugin" : {
                                                          "type" : "block",
                                                          "linehint" : 2000000,
                                                          "name" : "inputs/ssbm1000/part.csv",
                                                          "projections" : [ {
                                                            "relName" : "inputs/ssbm1000/part.csv",
                                                            "attrName" : "p_partkey"
                                                          }, {
                                                            "relName" : "inputs/ssbm1000/part.csv",
                                                            "attrName" : "p_category"
                                                          }, {
                                                            "relName" : "inputs/ssbm1000/part.csv",
                                                            "attrName" : "p_brand1"
                                                          } ],
                                                          "schema" : [ {
                                                            "relName" : "inputs/ssbm1000/part.csv",
                                                            "attrName" : "p_partkey",
                                                            "type" : {
                                                              "type" : "int"
                                                            },
                                                            "attrNo" : 1
                                                          }, {
                                                            "relName" : "inputs/ssbm1000/part.csv",
                                                            "attrName" : "p_name",
                                                            "type" : {
                                                              "type" : "dstring"
                                                            },
                                                            "attrNo" : 2
                                                          }, {
                                                            "relName" : "inputs/ssbm1000/part.csv",
                                                            "attrName" : "p_mfgr",
                                                            "type" : {
                                                              "type" : "dstring"
                                                            },
                                                            "attrNo" : 3
                                                          }, {
                                                            "relName" : "inputs/ssbm1000/part.csv",
                                                            "attrName" : "p_category",
                                                            "type" : {
                                                              "type" : "dstring"
                                                            },
                                                            "attrNo" : 4
                                                          }, {
                                                            "relName" : "inputs/ssbm1000/part.csv",
                                                            "attrName" : "p_brand1",
                                                            "type" : {
                                                              "type" : "dstring"
                                                            },
                                                            "attrNo" : 5
                                                          }, {
                                                            "relName" : "inputs/ssbm1000/part.csv",
                                                            "attrName" : "p_color",
                                                            "type" : {
                                                              "type" : "dstring"
                                                            },
                                                            "attrNo" : 6
                                                          }, {
                                                            "relName" : "inputs/ssbm1000/part.csv",
                                                            "attrName" : "p_type",
                                                            "type" : {
                                                              "type" : "dstring"
                                                            },
                                                            "attrNo" : 7
                                                          }, {
                                                            "relName" : "inputs/ssbm1000/part.csv",
                                                            "attrName" : "p_size",
                                                            "type" : {
                                                              "type" : "int"
                                                            },
                                                            "attrNo" : 8
                                                          }, {
                                                            "relName" : "inputs/ssbm1000/part.csv",
                                                            "attrName" : "p_container",
                                                            "type" : {
                                                              "type" : "dstring"
                                                            },
                                                            "attrNo" : 9
                                                          } ]
                                                        }
                                                      },
                                                      "to_cpu" : false
                                                    }
                                                  }
                                                }
                                              }
                                            }
                                          },
                                          "probe_k" : {
                                            "expression" : "recordProjection",
                                            "e" : {
                                              "expression" : "argument",
                                              "attributes" : [ {
                                                "attrName" : "lo_partkey",
                                                "relName" : "subsetPelagoProject#5474"
                                              } ],
                                              "type" : {
                                                "relName" : "subsetPelagoProject#5474",
                                                "type" : "record"
                                              },
                                              "argNo" : -1
                                            },
                                            "attribute" : {
                                              "attrName" : "lo_partkey",
                                              "relName" : "subsetPelagoProject#5474"
                                            },
                                            "register_as" : {
                                              "attrName" : "$0",
                                              "relName" : "subsetPelagoProject#5476"
                                            }
                                          },
                                          "probe_e" : [ {
                                            "e" : {
                                              "expression" : "recordProjection",
                                              "e" : {
                                                "expression" : "argument",
                                                "attributes" : [ {
                                                  "attrName" : "lo_custkey",
                                                  "relName" : "subsetPelagoProject#5474"
                                                } ],
                                                "type" : {
                                                  "relName" : "subsetPelagoProject#5474",
                                                  "type" : "record"
                                                },
                                                "argNo" : -1
                                              },
                                              "attribute" : {
                                                "attrName" : "lo_custkey",
                                                "relName" : "subsetPelagoProject#5474"
                                              },
                                              "register_as" : {
                                                "attrName" : "lo_custkey",
                                                "relName" : "subsetPelagoProject#5476"
                                              }
                                            },
                                            "packet" : 1,
                                            "offset" : 0
                                          }, {
                                            "e" : {
                                              "expression" : "recordProjection",
                                              "e" : {
                                                "expression" : "argument",
                                                "attributes" : [ {
                                                  "attrName" : "lo_partkey",
                                                  "relName" : "subsetPelagoProject#5474"
                                                } ],
                                                "type" : {
                                                  "relName" : "subsetPelagoProject#5474",
                                                  "type" : "record"
                                                },
                                                "argNo" : -1
                                              },
                                              "attribute" : {
                                                "attrName" : "lo_partkey",
                                                "relName" : "subsetPelagoProject#5474"
                                              },
                                              "register_as" : {
                                                "attrName" : "lo_partkey",
                                                "relName" : "subsetPelagoProject#5476"
                                              }
                                            },
                                            "packet" : 2,
                                            "offset" : 0
                                          }, {
                                            "e" : {
                                              "expression" : "recordProjection",
                                              "e" : {
                                                "expression" : "argument",
                                                "attributes" : [ {
                                                  "attrName" : "d_year",
                                                  "relName" : "subsetPelagoProject#5474"
                                                } ],
                                                "type" : {
                                                  "relName" : "subsetPelagoProject#5474",
                                                  "type" : "record"
                                                },
                                                "argNo" : -1
                                              },
                                              "attribute" : {
                                                "attrName" : "d_year",
                                                "relName" : "subsetPelagoProject#5474"
                                              },
                                              "register_as" : {
                                                "attrName" : "d_year",
                                                "relName" : "subsetPelagoProject#5476"
                                              }
                                            },
                                            "packet" : 3,
                                            "offset" : 0
                                          }, {
                                            "e" : {
                                              "expression" : "recordProjection",
                                              "e" : {
                                                "expression" : "argument",
                                                "attributes" : [ {
                                                  "attrName" : "s_city",
                                                  "relName" : "subsetPelagoProject#5474"
                                                } ],
                                                "type" : {
                                                  "relName" : "subsetPelagoProject#5474",
                                                  "type" : "record"
                                                },
                                                "argNo" : -1
                                              },
                                              "attribute" : {
                                                "attrName" : "s_city",
                                                "relName" : "subsetPelagoProject#5474"
                                              },
                                              "register_as" : {
                                                "attrName" : "s_city",
                                                "relName" : "subsetPelagoProject#5476"
                                              }
                                            },
                                            "packet" : 4,
                                            "offset" : 0
                                          }, {
                                            "e" : {
                                              "expression" : "recordProjection",
                                              "e" : {
                                                "expression" : "argument",
                                                "attributes" : [ {
                                                  "attrName" : "-",
                                                  "relName" : "subsetPelagoProject#5474"
                                                } ],
                                                "type" : {
                                                  "relName" : "subsetPelagoProject#5474",
                                                  "type" : "record"
                                                },
                                                "argNo" : -1
                                              },
                                              "attribute" : {
                                                "attrName" : "-",
                                                "relName" : "subsetPelagoProject#5474"
                                              },
                                              "register_as" : {
                                                "attrName" : "-",
                                                "relName" : "subsetPelagoProject#5476"
                                              }
                                            },
                                            "packet" : 5,
                                            "offset" : 0
                                          } ],
                                          "probe_w" : [ 64, 32, 32, 32, 32, 32 ],
                                          "hash_bits" : 20,
                                          "maxBuildInputSize" : 2000000,
                                          "probe_input" : {
                                            "operator" : "project",
                                            "gpu" : true,
                                            "relName" : "subsetPelagoProject#5474",
                                            "e" : [ {
                                              "expression" : "recordProjection",
                                              "e" : {
                                                "expression" : "argument",
                                                "attributes" : [ {
                                                  "attrName" : "lo_custkey",
                                                  "relName" : "subsetPelagoProject#5474"
                                                } ],
                                                "type" : {
                                                  "relName" : "subsetPelagoProject#5474",
                                                  "type" : "record"
                                                },
                                                "argNo" : -1
                                              },
                                              "attribute" : {
                                                "attrName" : "lo_custkey",
                                                "relName" : "subsetPelagoProject#5474"
                                              },
                                              "register_as" : {
                                                "attrName" : "lo_custkey",
                                                "relName" : "subsetPelagoProject#5474"
                                              }
                                            }, {
                                              "expression" : "recordProjection",
                                              "e" : {
                                                "expression" : "argument",
                                                "attributes" : [ {
                                                  "attrName" : "lo_partkey",
                                                  "relName" : "subsetPelagoProject#5474"
                                                } ],
                                                "type" : {
                                                  "relName" : "subsetPelagoProject#5474",
                                                  "type" : "record"
                                                },
                                                "argNo" : -1
                                              },
                                              "attribute" : {
                                                "attrName" : "lo_partkey",
                                                "relName" : "subsetPelagoProject#5474"
                                              },
                                              "register_as" : {
                                                "attrName" : "lo_partkey",
                                                "relName" : "subsetPelagoProject#5474"
                                              }
                                            }, {
                                              "expression" : "recordProjection",
                                              "e" : {
                                                "expression" : "argument",
                                                "attributes" : [ {
                                                  "attrName" : "d_year",
                                                  "relName" : "subsetPelagoProject#5474"
                                                } ],
                                                "type" : {
                                                  "relName" : "subsetPelagoProject#5474",
                                                  "type" : "record"
                                                },
                                                "argNo" : -1
                                              },
                                              "attribute" : {
                                                "attrName" : "d_year",
                                                "relName" : "subsetPelagoProject#5474"
                                              },
                                              "register_as" : {
                                                "attrName" : "d_year",
                                                "relName" : "subsetPelagoProject#5474"
                                              }
                                            }, {
                                              "expression" : "recordProjection",
                                              "e" : {
                                                "expression" : "argument",
                                                "attributes" : [ {
                                                  "attrName" : "s_city",
                                                  "relName" : "subsetPelagoProject#5474"
                                                } ],
                                                "type" : {
                                                  "relName" : "subsetPelagoProject#5474",
                                                  "type" : "record"
                                                },
                                                "argNo" : -1
                                              },
                                              "attribute" : {
                                                "attrName" : "s_city",
                                                "relName" : "subsetPelagoProject#5474"
                                              },
                                              "register_as" : {
                                                "attrName" : "s_city",
                                                "relName" : "subsetPelagoProject#5474"
                                              }
                                            }, {
                                              "expression" : "recordProjection",
                                              "e" : {
                                                "expression" : "argument",
                                                "attributes" : [ {
                                                  "attrName" : "-",
                                                  "relName" : "subsetPelagoProject#5474"
                                                } ],
                                                "type" : {
                                                  "relName" : "subsetPelagoProject#5474",
                                                  "type" : "record"
                                                },
                                                "argNo" : -1
                                              },
                                              "attribute" : {
                                                "attrName" : "-",
                                                "relName" : "subsetPelagoProject#5474"
                                              },
                                              "register_as" : {
                                                "attrName" : "-",
                                                "relName" : "subsetPelagoProject#5474"
                                              }
                                            } ],
                                            "input" : {
                                              "operator" : "hashjoin-chained",
                                              "gpu" : true,
                                              "build_k" : {
                                                "expression" : "recordProjection",
                                                "e" : {
                                                  "expression" : "argument",
                                                  "attributes" : [ {
                                                    "attrName" : "s_suppkey",
                                                    "relName" : "subsetPelagoProject#5462"
                                                  } ],
                                                  "type" : {
                                                    "relName" : "subsetPelagoProject#5462",
                                                    "type" : "record"
                                                  },
                                                  "argNo" : -1
                                                },
                                                "attribute" : {
                                                  "attrName" : "s_suppkey",
                                                  "relName" : "subsetPelagoProject#5462"
                                                },
                                                "register_as" : {
                                                  "attrName" : "$4",
                                                  "relName" : "subsetPelagoProject#5474"
                                                }
                                              },
                                              "build_e" : [ {
                                                "e" : {
                                                  "expression" : "recordProjection",
                                                  "e" : {
                                                    "expression" : "argument",
                                                    "attributes" : [ {
                                                      "attrName" : "s_suppkey",
                                                      "relName" : "subsetPelagoProject#5462"
                                                    } ],
                                                    "type" : {
                                                      "relName" : "subsetPelagoProject#5462",
                                                      "type" : "record"
                                                    },
                                                    "argNo" : -1
                                                  },
                                                  "attribute" : {
                                                    "attrName" : "s_suppkey",
                                                    "relName" : "subsetPelagoProject#5462"
                                                  },
                                                  "register_as" : {
                                                    "attrName" : "s_suppkey",
                                                    "relName" : "subsetPelagoProject#5474"
                                                  }
                                                },
                                                "packet" : 1,
                                                "offset" : 0
                                              }, {
                                                "e" : {
                                                  "expression" : "recordProjection",
                                                  "e" : {
                                                    "expression" : "argument",
                                                    "attributes" : [ {
                                                      "attrName" : "s_city",
                                                      "relName" : "subsetPelagoProject#5462"
                                                    } ],
                                                    "type" : {
                                                      "relName" : "subsetPelagoProject#5462",
                                                      "type" : "record"
                                                    },
                                                    "argNo" : -1
                                                  },
                                                  "attribute" : {
                                                    "attrName" : "s_city",
                                                    "relName" : "subsetPelagoProject#5462"
                                                  },
                                                  "register_as" : {
                                                    "attrName" : "s_city",
                                                    "relName" : "subsetPelagoProject#5474"
                                                  }
                                                },
                                                "packet" : 2,
                                                "offset" : 0
                                              } ],
                                              "build_w" : [ 64, 32, 32 ],
                                              "build_input" : {
                                                "operator" : "project",
                                                "gpu" : true,
                                                "relName" : "subsetPelagoProject#5462",
                                                "e" : [ {
                                                  "expression" : "recordProjection",
                                                  "e" : {
                                                    "expression" : "argument",
                                                    "attributes" : [ {
                                                      "attrName" : "s_suppkey",
                                                      "relName" : "inputs/ssbm1000/supplier.csv"
                                                    } ],
                                                    "type" : {
                                                      "relName" : "inputs/ssbm1000/supplier.csv",
                                                      "type" : "record"
                                                    },
                                                    "argNo" : -1
                                                  },
                                                  "attribute" : {
                                                    "attrName" : "s_suppkey",
                                                    "relName" : "inputs/ssbm1000/supplier.csv"
                                                  },
                                                  "register_as" : {
                                                    "attrName" : "s_suppkey",
                                                    "relName" : "subsetPelagoProject#5462"
                                                  }
                                                }, {
                                                  "expression" : "recordProjection",
                                                  "e" : {
                                                    "expression" : "argument",
                                                    "attributes" : [ {
                                                      "attrName" : "s_city",
                                                      "relName" : "inputs/ssbm1000/supplier.csv"
                                                    } ],
                                                    "type" : {
                                                      "relName" : "inputs/ssbm1000/supplier.csv",
                                                      "type" : "record"
                                                    },
                                                    "argNo" : -1
                                                  },
                                                  "attribute" : {
                                                    "attrName" : "s_city",
                                                    "relName" : "inputs/ssbm1000/supplier.csv"
                                                  },
                                                  "register_as" : {
                                                    "attrName" : "s_city",
                                                    "relName" : "subsetPelagoProject#5462"
                                                  }
                                                } ],
                                                "input" : {
                                                  "operator" : "select",
                                                  "gpu" : true,
                                                  "p" : {
                                                    "expression" : "eq",
                                                    "left" : {
                                                      "expression" : "recordProjection",
                                                      "e" : {
                                                        "expression" : "argument",
                                                        "attributes" : [ {
                                                          "attrName" : "s_nation",
                                                          "relName" : "inputs/ssbm1000/supplier.csv"
                                                        } ],
                                                        "type" : {
                                                          "relName" : "inputs/ssbm1000/supplier.csv",
                                                          "type" : "record"
                                                        },
                                                        "argNo" : -1
                                                      },
                                                      "attribute" : {
                                                        "attrName" : "s_nation",
                                                        "relName" : "inputs/ssbm1000/supplier.csv"
                                                      }
                                                    },
                                                    "right" : {
                                                      "expression" : "dstring",
                                                      "v" : "UNITED STATES",
                                                      "dict" : {
                                                        "path" : "inputs/ssbm1000/supplier.csv.ProjectedRelDataTypeField(#2: s_nation VARCHAR,null).dict"
                                                      }
                                                    }
                                                  },
                                                  "input" : {
                                                    "operator" : "unpack",
                                                    "gpu" : true,
                                                    "projections" : [ {
                                                      "expression" : "recordProjection",
                                                      "e" : {
                                                        "expression" : "argument",
                                                        "attributes" : [ {
                                                          "attrName" : "s_suppkey",
                                                          "relName" : "inputs/ssbm1000/supplier.csv"
                                                        } ],
                                                        "type" : {
                                                          "relName" : "inputs/ssbm1000/supplier.csv",
                                                          "type" : "record"
                                                        },
                                                        "argNo" : -1
                                                      },
                                                      "attribute" : {
                                                        "attrName" : "s_suppkey",
                                                        "relName" : "inputs/ssbm1000/supplier.csv"
                                                      }
                                                    }, {
                                                      "expression" : "recordProjection",
                                                      "e" : {
                                                        "expression" : "argument",
                                                        "attributes" : [ {
                                                          "attrName" : "s_city",
                                                          "relName" : "inputs/ssbm1000/supplier.csv"
                                                        } ],
                                                        "type" : {
                                                          "relName" : "inputs/ssbm1000/supplier.csv",
                                                          "type" : "record"
                                                        },
                                                        "argNo" : -1
                                                      },
                                                      "attribute" : {
                                                        "attrName" : "s_city",
                                                        "relName" : "inputs/ssbm1000/supplier.csv"
                                                      }
                                                    }, {
                                                      "expression" : "recordProjection",
                                                      "e" : {
                                                        "expression" : "argument",
                                                        "attributes" : [ {
                                                          "attrName" : "s_nation",
                                                          "relName" : "inputs/ssbm1000/supplier.csv"
                                                        } ],
                                                        "type" : {
                                                          "relName" : "inputs/ssbm1000/supplier.csv",
                                                          "type" : "record"
                                                        },
                                                        "argNo" : -1
                                                      },
                                                      "attribute" : {
                                                        "attrName" : "s_nation",
                                                        "relName" : "inputs/ssbm1000/supplier.csv"
                                                      }
                                                    } ],
                                                    "input" : {
                                                      "operator" : "cpu-to-gpu",
                                                      "projections" : [ {
                                                        "relName" : "inputs/ssbm1000/supplier.csv",
                                                        "attrName" : "s_suppkey",
                                                        "isBlock" : true
                                                      }, {
                                                        "relName" : "inputs/ssbm1000/supplier.csv",
                                                        "attrName" : "s_city",
                                                        "isBlock" : true
                                                      }, {
                                                        "relName" : "inputs/ssbm1000/supplier.csv",
                                                        "attrName" : "s_nation",
                                                        "isBlock" : true
                                                      } ],
                                                      "queueSize" : 262144,
                                                      "granularity" : "thread",
                                                      "input" : {
                                                        "operator" : "router",
                                                        "gpu" : false,
                                                        "projections" : [ {
                                                          "relName" : "inputs/ssbm1000/supplier.csv",
                                                          "attrName" : "s_suppkey",
                                                          "isBlock" : true
                                                        }, {
                                                          "relName" : "inputs/ssbm1000/supplier.csv",
                                                          "attrName" : "s_city",
                                                          "isBlock" : true
                                                        }, {
                                                          "relName" : "inputs/ssbm1000/supplier.csv",
                                                          "attrName" : "s_nation",
                                                          "isBlock" : true
                                                        } ],
                                                        "numOfParents" : 2,
                                                        "producers" : 1,
                                                        "slack" : 8,
                                                        "cpu_targets" : false,
                                                        "target" : {
                                                          "expression" : "recordProjection",
                                                          "e" : {
                                                            "expression" : "argument",
                                                            "argNo" : -1,
                                                            "type" : {
                                                              "type" : "record",
                                                              "relName" : "inputs/ssbm1000/supplier.csv"
                                                            },
                                                            "attributes" : [ {
                                                              "relName" : "inputs/ssbm1000/supplier.csv",
                                                              "attrName" : "__broadcastTarget"
                                                            } ]
                                                          },
                                                          "attribute" : {
                                                            "relName" : "inputs/ssbm1000/supplier.csv",
                                                            "attrName" : "__broadcastTarget"
                                                          }
                                                        },
                                                        "input" : {
                                                          "operator" : "mem-broadcast-device",
                                                          "num_of_targets" : 2,
                                                          "projections" : [ {
                                                            "relName" : "inputs/ssbm1000/supplier.csv",
                                                            "attrName" : "s_suppkey",
                                                            "isBlock" : true
                                                          }, {
                                                            "relName" : "inputs/ssbm1000/supplier.csv",
                                                            "attrName" : "s_city",
                                                            "isBlock" : true
                                                          }, {
                                                            "relName" : "inputs/ssbm1000/supplier.csv",
                                                            "attrName" : "s_nation",
                                                            "isBlock" : true
                                                          } ],
                                                          "input" : {
                                                            "operator" : "scan",
                                                            "gpu" : false,
                                                            "plugin" : {
                                                              "type" : "block",
                                                              "linehint" : 2000000,
                                                              "name" : "inputs/ssbm1000/supplier.csv",
                                                              "projections" : [ {
                                                                "relName" : "inputs/ssbm1000/supplier.csv",
                                                                "attrName" : "s_suppkey"
                                                              }, {
                                                                "relName" : "inputs/ssbm1000/supplier.csv",
                                                                "attrName" : "s_city"
                                                              }, {
                                                                "relName" : "inputs/ssbm1000/supplier.csv",
                                                                "attrName" : "s_nation"
                                                              } ],
                                                              "schema" : [ {
                                                                "relName" : "inputs/ssbm1000/supplier.csv",
                                                                "attrName" : "s_suppkey",
                                                                "type" : {
                                                                  "type" : "int"
                                                                },
                                                                "attrNo" : 1
                                                              }, {
                                                                "relName" : "inputs/ssbm1000/supplier.csv",
                                                                "attrName" : "s_name",
                                                                "type" : {
                                                                  "type" : "dstring"
                                                                },
                                                                "attrNo" : 2
                                                              }, {
                                                                "relName" : "inputs/ssbm1000/supplier.csv",
                                                                "attrName" : "s_address",
                                                                "type" : {
                                                                  "type" : "dstring"
                                                                },
                                                                "attrNo" : 3
                                                              }, {
                                                                "relName" : "inputs/ssbm1000/supplier.csv",
                                                                "attrName" : "s_city",
                                                                "type" : {
                                                                  "type" : "dstring"
                                                                },
                                                                "attrNo" : 4
                                                              }, {
                                                                "relName" : "inputs/ssbm1000/supplier.csv",
                                                                "attrName" : "s_nation",
                                                                "type" : {
                                                                  "type" : "dstring"
                                                                },
                                                                "attrNo" : 5
                                                              }, {
                                                                "relName" : "inputs/ssbm1000/supplier.csv",
                                                                "attrName" : "s_region",
                                                                "type" : {
                                                                  "type" : "dstring"
                                                                },
                                                                "attrNo" : 6
                                                              }, {
                                                                "relName" : "inputs/ssbm1000/supplier.csv",
                                                                "attrName" : "s_phone",
                                                                "type" : {
                                                                  "type" : "dstring"
                                                                },
                                                                "attrNo" : 7
                                                              } ]
                                                            }
                                                          },
                                                          "to_cpu" : false
                                                        }
                                                      }
                                                    }
                                                  }
                                                }
                                              },
                                              "probe_k" : {
                                                "expression" : "recordProjection",
                                                "e" : {
                                                  "expression" : "argument",
                                                  "attributes" : [ {
                                                    "attrName" : "lo_suppkey",
                                                    "relName" : "subsetPelagoProject#5472"
                                                  } ],
                                                  "type" : {
                                                    "relName" : "subsetPelagoProject#5472",
                                                    "type" : "record"
                                                  },
                                                  "argNo" : -1
                                                },
                                                "attribute" : {
                                                  "attrName" : "lo_suppkey",
                                                  "relName" : "subsetPelagoProject#5472"
                                                },
                                                "register_as" : {
                                                  "attrName" : "$0",
                                                  "relName" : "subsetPelagoProject#5474"
                                                }
                                              },
                                              "probe_e" : [ {
                                                "e" : {
                                                  "expression" : "recordProjection",
                                                  "e" : {
                                                    "expression" : "argument",
                                                    "attributes" : [ {
                                                      "attrName" : "lo_custkey",
                                                      "relName" : "subsetPelagoProject#5472"
                                                    } ],
                                                    "type" : {
                                                      "relName" : "subsetPelagoProject#5472",
                                                      "type" : "record"
                                                    },
                                                    "argNo" : -1
                                                  },
                                                  "attribute" : {
                                                    "attrName" : "lo_custkey",
                                                    "relName" : "subsetPelagoProject#5472"
                                                  },
                                                  "register_as" : {
                                                    "attrName" : "lo_custkey",
                                                    "relName" : "subsetPelagoProject#5474"
                                                  }
                                                },
                                                "packet" : 1,
                                                "offset" : 0
                                              }, {
                                                "e" : {
                                                  "expression" : "recordProjection",
                                                  "e" : {
                                                    "expression" : "argument",
                                                    "attributes" : [ {
                                                      "attrName" : "lo_partkey",
                                                      "relName" : "subsetPelagoProject#5472"
                                                    } ],
                                                    "type" : {
                                                      "relName" : "subsetPelagoProject#5472",
                                                      "type" : "record"
                                                    },
                                                    "argNo" : -1
                                                  },
                                                  "attribute" : {
                                                    "attrName" : "lo_partkey",
                                                    "relName" : "subsetPelagoProject#5472"
                                                  },
                                                  "register_as" : {
                                                    "attrName" : "lo_partkey",
                                                    "relName" : "subsetPelagoProject#5474"
                                                  }
                                                },
                                                "packet" : 2,
                                                "offset" : 0
                                              }, {
                                                "e" : {
                                                  "expression" : "recordProjection",
                                                  "e" : {
                                                    "expression" : "argument",
                                                    "attributes" : [ {
                                                      "attrName" : "lo_suppkey",
                                                      "relName" : "subsetPelagoProject#5472"
                                                    } ],
                                                    "type" : {
                                                      "relName" : "subsetPelagoProject#5472",
                                                      "type" : "record"
                                                    },
                                                    "argNo" : -1
                                                  },
                                                  "attribute" : {
                                                    "attrName" : "lo_suppkey",
                                                    "relName" : "subsetPelagoProject#5472"
                                                  },
                                                  "register_as" : {
                                                    "attrName" : "lo_suppkey",
                                                    "relName" : "subsetPelagoProject#5474"
                                                  }
                                                },
                                                "packet" : 3,
                                                "offset" : 0
                                              }, {
                                                "e" : {
                                                  "expression" : "recordProjection",
                                                  "e" : {
                                                    "expression" : "argument",
                                                    "attributes" : [ {
                                                      "attrName" : "d_year",
                                                      "relName" : "subsetPelagoProject#5472"
                                                    } ],
                                                    "type" : {
                                                      "relName" : "subsetPelagoProject#5472",
                                                      "type" : "record"
                                                    },
                                                    "argNo" : -1
                                                  },
                                                  "attribute" : {
                                                    "attrName" : "d_year",
                                                    "relName" : "subsetPelagoProject#5472"
                                                  },
                                                  "register_as" : {
                                                    "attrName" : "d_year",
                                                    "relName" : "subsetPelagoProject#5474"
                                                  }
                                                },
                                                "packet" : 4,
                                                "offset" : 0
                                              }, {
                                                "e" : {
                                                  "expression" : "recordProjection",
                                                  "e" : {
                                                    "expression" : "argument",
                                                    "attributes" : [ {
                                                      "attrName" : "-",
                                                      "relName" : "subsetPelagoProject#5472"
                                                    } ],
                                                    "type" : {
                                                      "relName" : "subsetPelagoProject#5472",
                                                      "type" : "record"
                                                    },
                                                    "argNo" : -1
                                                  },
                                                  "attribute" : {
                                                    "attrName" : "-",
                                                    "relName" : "subsetPelagoProject#5472"
                                                  },
                                                  "register_as" : {
                                                    "attrName" : "-",
                                                    "relName" : "subsetPelagoProject#5474"
                                                  }
                                                },
                                                "packet" : 5,
                                                "offset" : 0
                                              } ],
                                              "probe_w" : [ 64, 32, 32, 32, 32, 32 ],
                                              "hash_bits" : 12,
                                              "maxBuildInputSize" : 2000000,
                                              "probe_input" : {
                                                "operator" : "project",
                                                "gpu" : true,
                                                "relName" : "subsetPelagoProject#5472",
                                                "e" : [ {
                                                  "expression" : "recordProjection",
                                                  "e" : {
                                                    "expression" : "argument",
                                                    "attributes" : [ {
                                                      "attrName" : "lo_custkey",
                                                      "relName" : "subsetPelagoProject#5472"
                                                    } ],
                                                    "type" : {
                                                      "relName" : "subsetPelagoProject#5472",
                                                      "type" : "record"
                                                    },
                                                    "argNo" : -1
                                                  },
                                                  "attribute" : {
                                                    "attrName" : "lo_custkey",
                                                    "relName" : "subsetPelagoProject#5472"
                                                  },
                                                  "register_as" : {
                                                    "attrName" : "lo_custkey",
                                                    "relName" : "subsetPelagoProject#5472"
                                                  }
                                                }, {
                                                  "expression" : "recordProjection",
                                                  "e" : {
                                                    "expression" : "argument",
                                                    "attributes" : [ {
                                                      "attrName" : "lo_partkey",
                                                      "relName" : "subsetPelagoProject#5472"
                                                    } ],
                                                    "type" : {
                                                      "relName" : "subsetPelagoProject#5472",
                                                      "type" : "record"
                                                    },
                                                    "argNo" : -1
                                                  },
                                                  "attribute" : {
                                                    "attrName" : "lo_partkey",
                                                    "relName" : "subsetPelagoProject#5472"
                                                  },
                                                  "register_as" : {
                                                    "attrName" : "lo_partkey",
                                                    "relName" : "subsetPelagoProject#5472"
                                                  }
                                                }, {
                                                  "expression" : "recordProjection",
                                                  "e" : {
                                                    "expression" : "argument",
                                                    "attributes" : [ {
                                                      "attrName" : "lo_suppkey",
                                                      "relName" : "subsetPelagoProject#5472"
                                                    } ],
                                                    "type" : {
                                                      "relName" : "subsetPelagoProject#5472",
                                                      "type" : "record"
                                                    },
                                                    "argNo" : -1
                                                  },
                                                  "attribute" : {
                                                    "attrName" : "lo_suppkey",
                                                    "relName" : "subsetPelagoProject#5472"
                                                  },
                                                  "register_as" : {
                                                    "attrName" : "lo_suppkey",
                                                    "relName" : "subsetPelagoProject#5472"
                                                  }
                                                }, {
                                                  "expression" : "recordProjection",
                                                  "e" : {
                                                    "expression" : "argument",
                                                    "attributes" : [ {
                                                      "attrName" : "d_year",
                                                      "relName" : "subsetPelagoProject#5472"
                                                    } ],
                                                    "type" : {
                                                      "relName" : "subsetPelagoProject#5472",
                                                      "type" : "record"
                                                    },
                                                    "argNo" : -1
                                                  },
                                                  "attribute" : {
                                                    "attrName" : "d_year",
                                                    "relName" : "subsetPelagoProject#5472"
                                                  },
                                                  "register_as" : {
                                                    "attrName" : "d_year",
                                                    "relName" : "subsetPelagoProject#5472"
                                                  }
                                                }, {
                                                  "expression" : "recordProjection",
                                                  "e" : {
                                                    "expression" : "argument",
                                                    "attributes" : [ {
                                                      "attrName" : "-",
                                                      "relName" : "subsetPelagoProject#5472"
                                                    } ],
                                                    "type" : {
                                                      "relName" : "subsetPelagoProject#5472",
                                                      "type" : "record"
                                                    },
                                                    "argNo" : -1
                                                  },
                                                  "attribute" : {
                                                    "attrName" : "-",
                                                    "relName" : "subsetPelagoProject#5472"
                                                  },
                                                  "register_as" : {
                                                    "attrName" : "-",
                                                    "relName" : "subsetPelagoProject#5472"
                                                  }
                                                } ],
                                                "input" : {
                                                  "operator" : "hashjoin-chained",
                                                  "gpu" : true,
                                                  "build_k" : {
                                                    "expression" : "recordProjection",
                                                    "e" : {
                                                      "expression" : "argument",
                                                      "attributes" : [ {
                                                        "attrName" : "d_datekey",
                                                        "relName" : "inputs/ssbm1000/date.csv"
                                                      } ],
                                                      "type" : {
                                                        "relName" : "inputs/ssbm1000/date.csv",
                                                        "type" : "record"
                                                      },
                                                      "argNo" : -1
                                                    },
                                                    "attribute" : {
                                                      "attrName" : "d_datekey",
                                                      "relName" : "inputs/ssbm1000/date.csv"
                                                    },
                                                    "register_as" : {
                                                      "attrName" : "$5",
                                                      "relName" : "subsetPelagoProject#5472"
                                                    }
                                                  },
                                                  "build_e" : [ {
                                                    "e" : {
                                                      "expression" : "recordProjection",
                                                      "e" : {
                                                        "expression" : "argument",
                                                        "attributes" : [ {
                                                          "attrName" : "d_datekey",
                                                          "relName" : "inputs/ssbm1000/date.csv"
                                                        } ],
                                                        "type" : {
                                                          "relName" : "inputs/ssbm1000/date.csv",
                                                          "type" : "record"
                                                        },
                                                        "argNo" : -1
                                                      },
                                                      "attribute" : {
                                                        "attrName" : "d_datekey",
                                                        "relName" : "inputs/ssbm1000/date.csv"
                                                      },
                                                      "register_as" : {
                                                        "attrName" : "d_datekey",
                                                        "relName" : "subsetPelagoProject#5472"
                                                      }
                                                    },
                                                    "packet" : 1,
                                                    "offset" : 0
                                                  }, {
                                                    "e" : {
                                                      "expression" : "recordProjection",
                                                      "e" : {
                                                        "expression" : "argument",
                                                        "attributes" : [ {
                                                          "attrName" : "d_year",
                                                          "relName" : "inputs/ssbm1000/date.csv"
                                                        } ],
                                                        "type" : {
                                                          "relName" : "inputs/ssbm1000/date.csv",
                                                          "type" : "record"
                                                        },
                                                        "argNo" : -1
                                                      },
                                                      "attribute" : {
                                                        "attrName" : "d_year",
                                                        "relName" : "inputs/ssbm1000/date.csv"
                                                      },
                                                      "register_as" : {
                                                        "attrName" : "d_year",
                                                        "relName" : "subsetPelagoProject#5472"
                                                      }
                                                    },
                                                    "packet" : 2,
                                                    "offset" : 0
                                                  } ],
                                                  "build_w" : [ 64, 32, 32 ],
                                                  "build_input" : {
                                                    "operator" : "select",
                                                    "gpu" : true,
                                                    "p" : {
                                                      "expression" : "or",
                                                      "left" : {
                                                        "expression" : "eq",
                                                        "left" : {
                                                          "expression" : "recordProjection",
                                                          "e" : {
                                                            "expression" : "argument",
                                                            "attributes" : [ {
                                                              "attrName" : "d_year",
                                                              "relName" : "inputs/ssbm1000/date.csv"
                                                            } ],
                                                            "type" : {
                                                              "relName" : "inputs/ssbm1000/date.csv",
                                                              "type" : "record"
                                                            },
                                                            "argNo" : -1
                                                          },
                                                          "attribute" : {
                                                            "attrName" : "d_year",
                                                            "relName" : "inputs/ssbm1000/date.csv"
                                                          }
                                                        },
                                                        "right" : {
                                                          "expression" : "int",
                                                          "v" : 1997
                                                        }
                                                      },
                                                      "right" : {
                                                        "expression" : "eq",
                                                        "left" : {
                                                          "expression" : "recordProjection",
                                                          "e" : {
                                                            "expression" : "argument",
                                                            "attributes" : [ {
                                                              "attrName" : "d_year",
                                                              "relName" : "inputs/ssbm1000/date.csv"
                                                            } ],
                                                            "type" : {
                                                              "relName" : "inputs/ssbm1000/date.csv",
                                                              "type" : "record"
                                                            },
                                                            "argNo" : -1
                                                          },
                                                          "attribute" : {
                                                            "attrName" : "d_year",
                                                            "relName" : "inputs/ssbm1000/date.csv"
                                                          }
                                                        },
                                                        "right" : {
                                                          "expression" : "int",
                                                          "v" : 1998
                                                        }
                                                      }
                                                    },
                                                    "input" : {
                                                      "operator" : "unpack",
                                                      "gpu" : true,
                                                      "projections" : [ {
                                                        "expression" : "recordProjection",
                                                        "e" : {
                                                          "expression" : "argument",
                                                          "attributes" : [ {
                                                            "attrName" : "d_datekey",
                                                            "relName" : "inputs/ssbm1000/date.csv"
                                                          } ],
                                                          "type" : {
                                                            "relName" : "inputs/ssbm1000/date.csv",
                                                            "type" : "record"
                                                          },
                                                          "argNo" : -1
                                                        },
                                                        "attribute" : {
                                                          "attrName" : "d_datekey",
                                                          "relName" : "inputs/ssbm1000/date.csv"
                                                        }
                                                      }, {
                                                        "expression" : "recordProjection",
                                                        "e" : {
                                                          "expression" : "argument",
                                                          "attributes" : [ {
                                                            "attrName" : "d_year",
                                                            "relName" : "inputs/ssbm1000/date.csv"
                                                          } ],
                                                          "type" : {
                                                            "relName" : "inputs/ssbm1000/date.csv",
                                                            "type" : "record"
                                                          },
                                                          "argNo" : -1
                                                        },
                                                        "attribute" : {
                                                          "attrName" : "d_year",
                                                          "relName" : "inputs/ssbm1000/date.csv"
                                                        }
                                                      } ],
                                                      "input" : {
                                                        "operator" : "cpu-to-gpu",
                                                        "projections" : [ {
                                                          "relName" : "inputs/ssbm1000/date.csv",
                                                          "attrName" : "d_datekey",
                                                          "isBlock" : true
                                                        }, {
                                                          "relName" : "inputs/ssbm1000/date.csv",
                                                          "attrName" : "d_year",
                                                          "isBlock" : true
                                                        } ],
                                                        "queueSize" : 262144,
                                                        "granularity" : "thread",
                                                        "input" : {
                                                          "operator" : "router",
                                                          "gpu" : false,
                                                          "projections" : [ {
                                                            "relName" : "inputs/ssbm1000/date.csv",
                                                            "attrName" : "d_datekey",
                                                            "isBlock" : true
                                                          }, {
                                                            "relName" : "inputs/ssbm1000/date.csv",
                                                            "attrName" : "d_year",
                                                            "isBlock" : true
                                                          } ],
                                                          "numOfParents" : 2,
                                                          "producers" : 1,
                                                          "slack" : 8,
                                                          "cpu_targets" : false,
                                                          "target" : {
                                                            "expression" : "recordProjection",
                                                            "e" : {
                                                              "expression" : "argument",
                                                              "argNo" : -1,
                                                              "type" : {
                                                                "type" : "record",
                                                                "relName" : "inputs/ssbm1000/date.csv"
                                                              },
                                                              "attributes" : [ {
                                                                "relName" : "inputs/ssbm1000/date.csv",
                                                                "attrName" : "__broadcastTarget"
                                                              } ]
                                                            },
                                                            "attribute" : {
                                                              "relName" : "inputs/ssbm1000/date.csv",
                                                              "attrName" : "__broadcastTarget"
                                                            }
                                                          },
                                                          "input" : {
                                                            "operator" : "mem-broadcast-device",
                                                            "num_of_targets" : 2,
                                                            "projections" : [ {
                                                              "relName" : "inputs/ssbm1000/date.csv",
                                                              "attrName" : "d_datekey",
                                                              "isBlock" : true
                                                            }, {
                                                              "relName" : "inputs/ssbm1000/date.csv",
                                                              "attrName" : "d_year",
                                                              "isBlock" : true
                                                            } ],
                                                            "input" : {
                                                              "operator" : "scan",
                                                              "gpu" : false,
                                                              "plugin" : {
                                                                "type" : "block",
                                                                "linehint" : 2556,
                                                                "name" : "inputs/ssbm1000/date.csv",
                                                                "projections" : [ {
                                                                  "relName" : "inputs/ssbm1000/date.csv",
                                                                  "attrName" : "d_datekey"
                                                                }, {
                                                                  "relName" : "inputs/ssbm1000/date.csv",
                                                                  "attrName" : "d_year"
                                                                } ],
                                                                "schema" : [ {
                                                                  "relName" : "inputs/ssbm1000/date.csv",
                                                                  "attrName" : "d_datekey",
                                                                  "type" : {
                                                                    "type" : "int"
                                                                  },
                                                                  "attrNo" : 1
                                                                }, {
                                                                  "relName" : "inputs/ssbm1000/date.csv",
                                                                  "attrName" : "d_date",
                                                                  "type" : {
                                                                    "type" : "dstring"
                                                                  },
                                                                  "attrNo" : 2
                                                                }, {
                                                                  "relName" : "inputs/ssbm1000/date.csv",
                                                                  "attrName" : "d_dayofweek",
                                                                  "type" : {
                                                                    "type" : "dstring"
                                                                  },
                                                                  "attrNo" : 3
                                                                }, {
                                                                  "relName" : "inputs/ssbm1000/date.csv",
                                                                  "attrName" : "d_month",
                                                                  "type" : {
                                                                    "type" : "dstring"
                                                                  },
                                                                  "attrNo" : 4
                                                                }, {
                                                                  "relName" : "inputs/ssbm1000/date.csv",
                                                                  "attrName" : "d_year",
                                                                  "type" : {
                                                                    "type" : "int"
                                                                  },
                                                                  "attrNo" : 5
                                                                }, {
                                                                  "relName" : "inputs/ssbm1000/date.csv",
                                                                  "attrName" : "d_yearmonthnum",
                                                                  "type" : {
                                                                    "type" : "int"
                                                                  },
                                                                  "attrNo" : 6
                                                                }, {
                                                                  "relName" : "inputs/ssbm1000/date.csv",
                                                                  "attrName" : "d_yearmonth",
                                                                  "type" : {
                                                                    "type" : "dstring"
                                                                  },
                                                                  "attrNo" : 7
                                                                }, {
                                                                  "relName" : "inputs/ssbm1000/date.csv",
                                                                  "attrName" : "d_daynuminweek",
                                                                  "type" : {
                                                                    "type" : "int"
                                                                  },
                                                                  "attrNo" : 8
                                                                }, {
                                                                  "relName" : "inputs/ssbm1000/date.csv",
                                                                  "attrName" : "d_daynuminmonth",
                                                                  "type" : {
                                                                    "type" : "int"
                                                                  },
                                                                  "attrNo" : 9
                                                                }, {
                                                                  "relName" : "inputs/ssbm1000/date.csv",
                                                                  "attrName" : "d_daynuminyear",
                                                                  "type" : {
                                                                    "type" : "int"
                                                                  },
                                                                  "attrNo" : 10
                                                                }, {
                                                                  "relName" : "inputs/ssbm1000/date.csv",
                                                                  "attrName" : "d_monthnuminyear",
                                                                  "type" : {
                                                                    "type" : "int"
                                                                  },
                                                                  "attrNo" : 11
                                                                }, {
                                                                  "relName" : "inputs/ssbm1000/date.csv",
                                                                  "attrName" : "d_weeknuminyear",
                                                                  "type" : {
                                                                    "type" : "int"
                                                                  },
                                                                  "attrNo" : 12
                                                                }, {
                                                                  "relName" : "inputs/ssbm1000/date.csv",
                                                                  "attrName" : "d_sellingseason",
                                                                  "type" : {
                                                                    "type" : "dstring"
                                                                  },
                                                                  "attrNo" : 13
                                                                }, {
                                                                  "relName" : "inputs/ssbm1000/date.csv",
                                                                  "attrName" : "d_lastdayinweekfl",
                                                                  "type" : {
                                                                    "type" : "int"
                                                                  },
                                                                  "attrNo" : 14
                                                                }, {
                                                                  "relName" : "inputs/ssbm1000/date.csv",
                                                                  "attrName" : "d_lastdayinmonthfl",
                                                                  "type" : {
                                                                    "type" : "int"
                                                                  },
                                                                  "attrNo" : 15
                                                                }, {
                                                                  "relName" : "inputs/ssbm1000/date.csv",
                                                                  "attrName" : "d_holidayfl",
                                                                  "type" : {
                                                                    "type" : "int"
                                                                  },
                                                                  "attrNo" : 16
                                                                }, {
                                                                  "relName" : "inputs/ssbm1000/date.csv",
                                                                  "attrName" : "d_weekdayfl",
                                                                  "type" : {
                                                                    "type" : "int"
                                                                  },
                                                                  "attrNo" : 17
                                                                } ]
                                                              }
                                                            },
                                                            "to_cpu" : false
                                                          }
                                                        }
                                                      }
                                                    }
                                                  },
                                                  "probe_k" : {
                                                    "expression" : "recordProjection",
                                                    "e" : {
                                                      "expression" : "argument",
                                                      "attributes" : [ {
                                                        "attrName" : "lo_orderdate",
                                                        "relName" : "subsetPelagoProject#5470"
                                                      } ],
                                                      "type" : {
                                                        "relName" : "subsetPelagoProject#5470",
                                                        "type" : "record"
                                                      },
                                                      "argNo" : -1
                                                    },
                                                    "attribute" : {
                                                      "attrName" : "lo_orderdate",
                                                      "relName" : "subsetPelagoProject#5470"
                                                    },
                                                    "register_as" : {
                                                      "attrName" : "$0",
                                                      "relName" : "subsetPelagoProject#5472"
                                                    }
                                                  },
                                                  "probe_e" : [ {
                                                    "e" : {
                                                      "expression" : "recordProjection",
                                                      "e" : {
                                                        "expression" : "argument",
                                                        "attributes" : [ {
                                                          "attrName" : "lo_custkey",
                                                          "relName" : "subsetPelagoProject#5470"
                                                        } ],
                                                        "type" : {
                                                          "relName" : "subsetPelagoProject#5470",
                                                          "type" : "record"
                                                        },
                                                        "argNo" : -1
                                                      },
                                                      "attribute" : {
                                                        "attrName" : "lo_custkey",
                                                        "relName" : "subsetPelagoProject#5470"
                                                      },
                                                      "register_as" : {
                                                        "attrName" : "lo_custkey",
                                                        "relName" : "subsetPelagoProject#5472"
                                                      }
                                                    },
                                                    "packet" : 1,
                                                    "offset" : 0
                                                  }, {
                                                    "e" : {
                                                      "expression" : "recordProjection",
                                                      "e" : {
                                                        "expression" : "argument",
                                                        "attributes" : [ {
                                                          "attrName" : "lo_partkey",
                                                          "relName" : "subsetPelagoProject#5470"
                                                        } ],
                                                        "type" : {
                                                          "relName" : "subsetPelagoProject#5470",
                                                          "type" : "record"
                                                        },
                                                        "argNo" : -1
                                                      },
                                                      "attribute" : {
                                                        "attrName" : "lo_partkey",
                                                        "relName" : "subsetPelagoProject#5470"
                                                      },
                                                      "register_as" : {
                                                        "attrName" : "lo_partkey",
                                                        "relName" : "subsetPelagoProject#5472"
                                                      }
                                                    },
                                                    "packet" : 2,
                                                    "offset" : 0
                                                  }, {
                                                    "e" : {
                                                      "expression" : "recordProjection",
                                                      "e" : {
                                                        "expression" : "argument",
                                                        "attributes" : [ {
                                                          "attrName" : "lo_suppkey",
                                                          "relName" : "subsetPelagoProject#5470"
                                                        } ],
                                                        "type" : {
                                                          "relName" : "subsetPelagoProject#5470",
                                                          "type" : "record"
                                                        },
                                                        "argNo" : -1
                                                      },
                                                      "attribute" : {
                                                        "attrName" : "lo_suppkey",
                                                        "relName" : "subsetPelagoProject#5470"
                                                      },
                                                      "register_as" : {
                                                        "attrName" : "lo_suppkey",
                                                        "relName" : "subsetPelagoProject#5472"
                                                      }
                                                    },
                                                    "packet" : 3,
                                                    "offset" : 0
                                                  }, {
                                                    "e" : {
                                                      "expression" : "recordProjection",
                                                      "e" : {
                                                        "expression" : "argument",
                                                        "attributes" : [ {
                                                          "attrName" : "lo_orderdate",
                                                          "relName" : "subsetPelagoProject#5470"
                                                        } ],
                                                        "type" : {
                                                          "relName" : "subsetPelagoProject#5470",
                                                          "type" : "record"
                                                        },
                                                        "argNo" : -1
                                                      },
                                                      "attribute" : {
                                                        "attrName" : "lo_orderdate",
                                                        "relName" : "subsetPelagoProject#5470"
                                                      },
                                                      "register_as" : {
                                                        "attrName" : "lo_orderdate",
                                                        "relName" : "subsetPelagoProject#5472"
                                                      }
                                                    },
                                                    "packet" : 4,
                                                    "offset" : 0
                                                  }, {
                                                    "e" : {
                                                      "expression" : "recordProjection",
                                                      "e" : {
                                                        "expression" : "argument",
                                                        "attributes" : [ {
                                                          "attrName" : "-",
                                                          "relName" : "subsetPelagoProject#5470"
                                                        } ],
                                                        "type" : {
                                                          "relName" : "subsetPelagoProject#5470",
                                                          "type" : "record"
                                                        },
                                                        "argNo" : -1
                                                      },
                                                      "attribute" : {
                                                        "attrName" : "-",
                                                        "relName" : "subsetPelagoProject#5470"
                                                      },
                                                      "register_as" : {
                                                        "attrName" : "-",
                                                        "relName" : "subsetPelagoProject#5472"
                                                      }
                                                    },
                                                    "packet" : 5,
                                                    "offset" : 0
                                                  } ],
                                                  "probe_w" : [ 64, 32, 32, 32, 32, 32 ],
                                                  "hash_bits" : 13,
                                                  "maxBuildInputSize" : 2556,
                                                  "probe_input" : {
                                                    "operator" : "project",
                                                    "gpu" : true,
                                                    "relName" : "subsetPelagoProject#5470",
                                                    "e" : [ {
                                                      "expression" : "recordProjection",
                                                      "e" : {
                                                        "expression" : "argument",
                                                        "attributes" : [ {
                                                          "attrName" : "lo_custkey",
                                                          "relName" : "inputs/ssbm1000/lineorder.csv"
                                                        } ],
                                                        "type" : {
                                                          "relName" : "inputs/ssbm1000/lineorder.csv",
                                                          "type" : "record"
                                                        },
                                                        "argNo" : -1
                                                      },
                                                      "attribute" : {
                                                        "attrName" : "lo_custkey",
                                                        "relName" : "inputs/ssbm1000/lineorder.csv"
                                                      },
                                                      "register_as" : {
                                                        "attrName" : "lo_custkey",
                                                        "relName" : "subsetPelagoProject#5470"
                                                      }
                                                    }, {
                                                      "expression" : "recordProjection",
                                                      "e" : {
                                                        "expression" : "argument",
                                                        "attributes" : [ {
                                                          "attrName" : "lo_partkey",
                                                          "relName" : "inputs/ssbm1000/lineorder.csv"
                                                        } ],
                                                        "type" : {
                                                          "relName" : "inputs/ssbm1000/lineorder.csv",
                                                          "type" : "record"
                                                        },
                                                        "argNo" : -1
                                                      },
                                                      "attribute" : {
                                                        "attrName" : "lo_partkey",
                                                        "relName" : "inputs/ssbm1000/lineorder.csv"
                                                      },
                                                      "register_as" : {
                                                        "attrName" : "lo_partkey",
                                                        "relName" : "subsetPelagoProject#5470"
                                                      }
                                                    }, {
                                                      "expression" : "recordProjection",
                                                      "e" : {
                                                        "expression" : "argument",
                                                        "attributes" : [ {
                                                          "attrName" : "lo_suppkey",
                                                          "relName" : "inputs/ssbm1000/lineorder.csv"
                                                        } ],
                                                        "type" : {
                                                          "relName" : "inputs/ssbm1000/lineorder.csv",
                                                          "type" : "record"
                                                        },
                                                        "argNo" : -1
                                                      },
                                                      "attribute" : {
                                                        "attrName" : "lo_suppkey",
                                                        "relName" : "inputs/ssbm1000/lineorder.csv"
                                                      },
                                                      "register_as" : {
                                                        "attrName" : "lo_suppkey",
                                                        "relName" : "subsetPelagoProject#5470"
                                                      }
                                                    }, {
                                                      "expression" : "recordProjection",
                                                      "e" : {
                                                        "expression" : "argument",
                                                        "attributes" : [ {
                                                          "attrName" : "lo_orderdate",
                                                          "relName" : "inputs/ssbm1000/lineorder.csv"
                                                        } ],
                                                        "type" : {
                                                          "relName" : "inputs/ssbm1000/lineorder.csv",
                                                          "type" : "record"
                                                        },
                                                        "argNo" : -1
                                                      },
                                                      "attribute" : {
                                                        "attrName" : "lo_orderdate",
                                                        "relName" : "inputs/ssbm1000/lineorder.csv"
                                                      },
                                                      "register_as" : {
                                                        "attrName" : "lo_orderdate",
                                                        "relName" : "subsetPelagoProject#5470"
                                                      }
                                                    }, {
                                                      "expression" : "sub",
                                                      "left" : {
                                                        "expression" : "recordProjection",
                                                        "e" : {
                                                          "expression" : "argument",
                                                          "attributes" : [ {
                                                            "attrName" : "lo_revenue",
                                                            "relName" : "inputs/ssbm1000/lineorder.csv"
                                                          } ],
                                                          "type" : {
                                                            "relName" : "inputs/ssbm1000/lineorder.csv",
                                                            "type" : "record"
                                                          },
                                                          "argNo" : -1
                                                        },
                                                        "attribute" : {
                                                          "attrName" : "lo_revenue",
                                                          "relName" : "inputs/ssbm1000/lineorder.csv"
                                                        }
                                                      },
                                                      "right" : {
                                                        "expression" : "recordProjection",
                                                        "e" : {
                                                          "expression" : "argument",
                                                          "attributes" : [ {
                                                            "attrName" : "lo_supplycost",
                                                            "relName" : "inputs/ssbm1000/lineorder.csv"
                                                          } ],
                                                          "type" : {
                                                            "relName" : "inputs/ssbm1000/lineorder.csv",
                                                            "type" : "record"
                                                          },
                                                          "argNo" : -1
                                                        },
                                                        "attribute" : {
                                                          "attrName" : "lo_supplycost",
                                                          "relName" : "inputs/ssbm1000/lineorder.csv"
                                                        }
                                                      },
                                                      "register_as" : {
                                                        "attrName" : "-",
                                                        "relName" : "subsetPelagoProject#5470"
                                                      }
                                                    } ],
                                                    "input" : {
                                                      "operator" : "unpack",
                                                      "gpu" : true,
                                                      "projections" : [ {
                                                        "expression" : "recordProjection",
                                                        "e" : {
                                                          "expression" : "argument",
                                                          "attributes" : [ {
                                                            "attrName" : "lo_custkey",
                                                            "relName" : "inputs/ssbm1000/lineorder.csv"
                                                          } ],
                                                          "type" : {
                                                            "relName" : "inputs/ssbm1000/lineorder.csv",
                                                            "type" : "record"
                                                          },
                                                          "argNo" : -1
                                                        },
                                                        "attribute" : {
                                                          "attrName" : "lo_custkey",
                                                          "relName" : "inputs/ssbm1000/lineorder.csv"
                                                        }
                                                      }, {
                                                        "expression" : "recordProjection",
                                                        "e" : {
                                                          "expression" : "argument",
                                                          "attributes" : [ {
                                                            "attrName" : "lo_partkey",
                                                            "relName" : "inputs/ssbm1000/lineorder.csv"
                                                          } ],
                                                          "type" : {
                                                            "relName" : "inputs/ssbm1000/lineorder.csv",
                                                            "type" : "record"
                                                          },
                                                          "argNo" : -1
                                                        },
                                                        "attribute" : {
                                                          "attrName" : "lo_partkey",
                                                          "relName" : "inputs/ssbm1000/lineorder.csv"
                                                        }
                                                      }, {
                                                        "expression" : "recordProjection",
                                                        "e" : {
                                                          "expression" : "argument",
                                                          "attributes" : [ {
                                                            "attrName" : "lo_suppkey",
                                                            "relName" : "inputs/ssbm1000/lineorder.csv"
                                                          } ],
                                                          "type" : {
                                                            "relName" : "inputs/ssbm1000/lineorder.csv",
                                                            "type" : "record"
                                                          },
                                                          "argNo" : -1
                                                        },
                                                        "attribute" : {
                                                          "attrName" : "lo_suppkey",
                                                          "relName" : "inputs/ssbm1000/lineorder.csv"
                                                        }
                                                      }, {
                                                        "expression" : "recordProjection",
                                                        "e" : {
                                                          "expression" : "argument",
                                                          "attributes" : [ {
                                                            "attrName" : "lo_orderdate",
                                                            "relName" : "inputs/ssbm1000/lineorder.csv"
                                                          } ],
                                                          "type" : {
                                                            "relName" : "inputs/ssbm1000/lineorder.csv",
                                                            "type" : "record"
                                                          },
                                                          "argNo" : -1
                                                        },
                                                        "attribute" : {
                                                          "attrName" : "lo_orderdate",
                                                          "relName" : "inputs/ssbm1000/lineorder.csv"
                                                        }
                                                      }, {
                                                        "expression" : "recordProjection",
                                                        "e" : {
                                                          "expression" : "argument",
                                                          "attributes" : [ {
                                                            "attrName" : "lo_revenue",
                                                            "relName" : "inputs/ssbm1000/lineorder.csv"
                                                          } ],
                                                          "type" : {
                                                            "relName" : "inputs/ssbm1000/lineorder.csv",
                                                            "type" : "record"
                                                          },
                                                          "argNo" : -1
                                                        },
                                                        "attribute" : {
                                                          "attrName" : "lo_revenue",
                                                          "relName" : "inputs/ssbm1000/lineorder.csv"
                                                        }
                                                      }, {
                                                        "expression" : "recordProjection",
                                                        "e" : {
                                                          "expression" : "argument",
                                                          "attributes" : [ {
                                                            "attrName" : "lo_supplycost",
                                                            "relName" : "inputs/ssbm1000/lineorder.csv"
                                                          } ],
                                                          "type" : {
                                                            "relName" : "inputs/ssbm1000/lineorder.csv",
                                                            "type" : "record"
                                                          },
                                                          "argNo" : -1
                                                        },
                                                        "attribute" : {
                                                          "attrName" : "lo_supplycost",
                                                          "relName" : "inputs/ssbm1000/lineorder.csv"
                                                        }
                                                      } ],
                                                      "input" : {
                                                        "operator" : "cpu-to-gpu",
                                                        "projections" : [ {
                                                          "relName" : "inputs/ssbm1000/lineorder.csv",
                                                          "attrName" : "lo_custkey",
                                                          "isBlock" : true
                                                        }, {
                                                          "relName" : "inputs/ssbm1000/lineorder.csv",
                                                          "attrName" : "lo_partkey",
                                                          "isBlock" : true
                                                        }, {
                                                          "relName" : "inputs/ssbm1000/lineorder.csv",
                                                          "attrName" : "lo_suppkey",
                                                          "isBlock" : true
                                                        }, {
                                                          "relName" : "inputs/ssbm1000/lineorder.csv",
                                                          "attrName" : "lo_orderdate",
                                                          "isBlock" : true
                                                        }, {
                                                          "relName" : "inputs/ssbm1000/lineorder.csv",
                                                          "attrName" : "lo_revenue",
                                                          "isBlock" : true
                                                        }, {
                                                          "relName" : "inputs/ssbm1000/lineorder.csv",
                                                          "attrName" : "lo_supplycost",
                                                          "isBlock" : true
                                                        } ],
                                                        "queueSize" : 262144,
                                                        "granularity" : "thread",
                                                        "input" : {
                                                          "operator" : "mem-move-device",
                                                          "projections" : [ {
                                                            "relName" : "inputs/ssbm1000/lineorder.csv",
                                                            "attrName" : "lo_custkey",
                                                            "isBlock" : true
                                                          }, {
                                                            "relName" : "inputs/ssbm1000/lineorder.csv",
                                                            "attrName" : "lo_partkey",
                                                            "isBlock" : true
                                                          }, {
                                                            "relName" : "inputs/ssbm1000/lineorder.csv",
                                                            "attrName" : "lo_suppkey",
                                                            "isBlock" : true
                                                          }, {
                                                            "relName" : "inputs/ssbm1000/lineorder.csv",
                                                            "attrName" : "lo_orderdate",
                                                            "isBlock" : true
                                                          }, {
                                                            "relName" : "inputs/ssbm1000/lineorder.csv",
                                                            "attrName" : "lo_revenue",
                                                            "isBlock" : true
                                                          }, {
                                                            "relName" : "inputs/ssbm1000/lineorder.csv",
                                                            "attrName" : "lo_supplycost",
                                                            "isBlock" : true
                                                          } ],
                                                          "input" : {
                                                            "operator" : "router",
                                                            "gpu" : false,
                                                            "projections" : [ {
                                                              "relName" : "inputs/ssbm1000/lineorder.csv",
                                                              "attrName" : "lo_custkey",
                                                              "isBlock" : true
                                                            }, {
                                                              "relName" : "inputs/ssbm1000/lineorder.csv",
                                                              "attrName" : "lo_partkey",
                                                              "isBlock" : true
                                                            }, {
                                                              "relName" : "inputs/ssbm1000/lineorder.csv",
                                                              "attrName" : "lo_suppkey",
                                                              "isBlock" : true
                                                            }, {
                                                              "relName" : "inputs/ssbm1000/lineorder.csv",
                                                              "attrName" : "lo_orderdate",
                                                              "isBlock" : true
                                                            }, {
                                                              "relName" : "inputs/ssbm1000/lineorder.csv",
                                                              "attrName" : "lo_revenue",
                                                              "isBlock" : true
                                                            }, {
                                                              "relName" : "inputs/ssbm1000/lineorder.csv",
                                                              "attrName" : "lo_supplycost",
                                                              "isBlock" : true
                                                            } ],
                                                            "numOfParents" : 2,
                                                            "producers" : 1,
                                                            "slack" : 8,
                                                            "cpu_targets" : false,
                                                            "numa_local" : true,
                                                            "input" : {
                                                              "operator" : "scan",
                                                              "gpu" : false,
                                                              "plugin" : {
                                                                "type" : "block",
                                                                "linehint" : 5999989813,
                                                                "name" : "inputs/ssbm1000/lineorder.csv",
                                                                "projections" : [ {
                                                                  "relName" : "inputs/ssbm1000/lineorder.csv",
                                                                  "attrName" : "lo_custkey"
                                                                }, {
                                                                  "relName" : "inputs/ssbm1000/lineorder.csv",
                                                                  "attrName" : "lo_partkey"
                                                                }, {
                                                                  "relName" : "inputs/ssbm1000/lineorder.csv",
                                                                  "attrName" : "lo_suppkey"
                                                                }, {
                                                                  "relName" : "inputs/ssbm1000/lineorder.csv",
                                                                  "attrName" : "lo_orderdate"
                                                                }, {
                                                                  "relName" : "inputs/ssbm1000/lineorder.csv",
                                                                  "attrName" : "lo_revenue"
                                                                }, {
                                                                  "relName" : "inputs/ssbm1000/lineorder.csv",
                                                                  "attrName" : "lo_supplycost"
                                                                } ],
                                                                "schema" : [ {
                                                                  "relName" : "inputs/ssbm1000/lineorder.csv",
                                                                  "attrName" : "lo_orderkey",
                                                                  "type" : {
                                                                    "type" : "int"
                                                                  },
                                                                  "attrNo" : 1
                                                                }, {
                                                                  "relName" : "inputs/ssbm1000/lineorder.csv",
                                                                  "attrName" : "lo_linenumber",
                                                                  "type" : {
                                                                    "type" : "int"
                                                                  },
                                                                  "attrNo" : 2
                                                                }, {
                                                                  "relName" : "inputs/ssbm1000/lineorder.csv",
                                                                  "attrName" : "lo_custkey",
                                                                  "type" : {
                                                                    "type" : "int"
                                                                  },
                                                                  "attrNo" : 3
                                                                }, {
                                                                  "relName" : "inputs/ssbm1000/lineorder.csv",
                                                                  "attrName" : "lo_partkey",
                                                                  "type" : {
                                                                    "type" : "int"
                                                                  },
                                                                  "attrNo" : 4
                                                                }, {
                                                                  "relName" : "inputs/ssbm1000/lineorder.csv",
                                                                  "attrName" : "lo_suppkey",
                                                                  "type" : {
                                                                    "type" : "int"
                                                                  },
                                                                  "attrNo" : 5
                                                                }, {
                                                                  "relName" : "inputs/ssbm1000/lineorder.csv",
                                                                  "attrName" : "lo_orderdate",
                                                                  "type" : {
                                                                    "type" : "int"
                                                                  },
                                                                  "attrNo" : 6
                                                                }, {
                                                                  "relName" : "inputs/ssbm1000/lineorder.csv",
                                                                  "attrName" : "lo_orderpriority",
                                                                  "type" : {
                                                                    "type" : "dstring"
                                                                  },
                                                                  "attrNo" : 7
                                                                }, {
                                                                  "relName" : "inputs/ssbm1000/lineorder.csv",
                                                                  "attrName" : "lo_shippriority",
                                                                  "type" : {
                                                                    "type" : "dstring"
                                                                  },
                                                                  "attrNo" : 8
                                                                }, {
                                                                  "relName" : "inputs/ssbm1000/lineorder.csv",
                                                                  "attrName" : "lo_quantity",
                                                                  "type" : {
                                                                    "type" : "int"
                                                                  },
                                                                  "attrNo" : 9
                                                                }, {
                                                                  "relName" : "inputs/ssbm1000/lineorder.csv",
                                                                  "attrName" : "lo_extendedprice",
                                                                  "type" : {
                                                                    "type" : "int"
                                                                  },
                                                                  "attrNo" : 10
                                                                }, {
                                                                  "relName" : "inputs/ssbm1000/lineorder.csv",
                                                                  "attrName" : "lo_ordtotalprice",
                                                                  "type" : {
                                                                    "type" : "int"
                                                                  },
                                                                  "attrNo" : 11
                                                                }, {
                                                                  "relName" : "inputs/ssbm1000/lineorder.csv",
                                                                  "attrName" : "lo_discount",
                                                                  "type" : {
                                                                    "type" : "int"
                                                                  },
                                                                  "attrNo" : 12
                                                                }, {
                                                                  "relName" : "inputs/ssbm1000/lineorder.csv",
                                                                  "attrName" : "lo_revenue",
                                                                  "type" : {
                                                                    "type" : "int"
                                                                  },
                                                                  "attrNo" : 13
                                                                }, {
                                                                  "relName" : "inputs/ssbm1000/lineorder.csv",
                                                                  "attrName" : "lo_supplycost",
                                                                  "type" : {
                                                                    "type" : "int"
                                                                  },
                                                                  "attrNo" : 14
                                                                }, {
                                                                  "relName" : "inputs/ssbm1000/lineorder.csv",
                                                                  "attrName" : "lo_tax",
                                                                  "type" : {
                                                                    "type" : "int"
                                                                  },
                                                                  "attrNo" : 15
                                                                }, {
                                                                  "relName" : "inputs/ssbm1000/lineorder.csv",
                                                                  "attrName" : "lo_commitdate",
                                                                  "type" : {
                                                                    "type" : "int"
                                                                  },
                                                                  "attrNo" : 16
                                                                }, {
                                                                  "relName" : "inputs/ssbm1000/lineorder.csv",
                                                                  "attrName" : "lo_shipmode",
                                                                  "type" : {
                                                                    "type" : "dstring"
                                                                  },
                                                                  "attrNo" : 17
                                                                } ]
                                                              }
                                                            }
                                                          },
                                                          "to_cpu" : false
                                                        }
                                                      }
                                                    }
                                                  }
                                                }
                                              }
                                            }
                                          }
                                        }
                                      }
                                    }
                                  }
                                },
                                "trait" : "Pelago.[].packed.NVPTX.homRandom.hetSingle"
                              }
                            },
                            "to_cpu" : false
                          }
                        },
                        "to_cpu" : false
                      }
                    }
                  }
                },
                "trait" : "Pelago.[].packed.NVPTX.homSingle.hetSingle"
              }
            },
            "to_cpu" : true
          }
        }
      }
    }
  }
}