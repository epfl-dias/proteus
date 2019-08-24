{
  "operator" : "print",
  "gpu" : false,
  "e" : [ {
    "expression" : "recordProjection",
    "e" : {
      "expression" : "argument",
      "attributes" : [ {
        "attrName" : "lo_revenue",
        "relName" : "subsetPelagoSort#11110"
      } ],
      "type" : {
        "relName" : "subsetPelagoSort#11110",
        "type" : "record"
      },
      "argNo" : -1
    },
    "attribute" : {
      "attrName" : "lo_revenue",
      "relName" : "subsetPelagoSort#11110"
    },
    "register_as" : {
      "attrName" : "lo_revenue",
      "relName" : "print11111"
    }
  }, {
    "expression" : "recordProjection",
    "e" : {
      "expression" : "argument",
      "attributes" : [ {
        "attrName" : "d_year",
        "relName" : "subsetPelagoSort#11110"
      } ],
      "type" : {
        "relName" : "subsetPelagoSort#11110",
        "type" : "record"
      },
      "argNo" : -1
    },
    "attribute" : {
      "attrName" : "d_year",
      "relName" : "subsetPelagoSort#11110"
    },
    "register_as" : {
      "attrName" : "d_year",
      "relName" : "print11111"
    }
  }, {
    "expression" : "recordProjection",
    "e" : {
      "expression" : "argument",
      "attributes" : [ {
        "attrName" : "p_brand1",
        "relName" : "subsetPelagoSort#11110"
      } ],
      "type" : {
        "relName" : "subsetPelagoSort#11110",
        "type" : "record"
      },
      "argNo" : -1
    },
    "attribute" : {
      "attrName" : "p_brand1",
      "relName" : "subsetPelagoSort#11110"
    },
    "register_as" : {
      "attrName" : "p_brand1",
      "relName" : "print11111"
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
            "relName" : "__sort11110"
          } ],
          "expression" : "argument",
          "type" : {
            "type" : "record",
            "relName" : "__sort11110"
          },
          "argNo" : 1
        },
        "expression" : "recordProjection",
        "attribute" : {
          "attrName" : "__sorted",
          "relName" : "__sort11110"
        }
      },
      "expression" : "recordProjection",
      "attribute" : {
        "attrName" : "lo_revenue",
        "relName" : "__sort11110"
      },
      "register_as" : {
        "attrName" : "lo_revenue",
        "relName" : "subsetPelagoSort#11110"
      }
    }, {
      "e" : {
        "e" : {
          "attributes" : [ {
            "attrName" : "__sorted",
            "relName" : "__sort11110"
          } ],
          "expression" : "argument",
          "type" : {
            "type" : "record",
            "relName" : "__sort11110"
          },
          "argNo" : 1
        },
        "expression" : "recordProjection",
        "attribute" : {
          "attrName" : "__sorted",
          "relName" : "__sort11110"
        }
      },
      "expression" : "recordProjection",
      "attribute" : {
        "attrName" : "d_year",
        "relName" : "__sort11110"
      },
      "register_as" : {
        "attrName" : "d_year",
        "relName" : "subsetPelagoSort#11110"
      }
    }, {
      "e" : {
        "e" : {
          "attributes" : [ {
            "attrName" : "__sorted",
            "relName" : "__sort11110"
          } ],
          "expression" : "argument",
          "type" : {
            "type" : "record",
            "relName" : "__sort11110"
          },
          "argNo" : 1
        },
        "expression" : "recordProjection",
        "attribute" : {
          "attrName" : "__sorted",
          "relName" : "__sort11110"
        }
      },
      "expression" : "recordProjection",
      "attribute" : {
        "attrName" : "p_brand1",
        "relName" : "__sort11110"
      },
      "register_as" : {
        "attrName" : "p_brand1",
        "relName" : "subsetPelagoSort#11110"
      }
    } ],
    "relName" : "subsetPelagoSort#11110",
    "input" : {
      "operator" : "unpack",
      "gpu" : false,
      "projections" : [ {
        "e" : {
          "attributes" : [ {
            "attrName" : "__sorted",
            "relName" : "__sort11110"
          } ],
          "expression" : "argument",
          "type" : {
            "type" : "record",
            "relName" : "__sort11110"
          },
          "argNo" : 1
        },
        "expression" : "recordProjection",
        "attribute" : {
          "attrName" : "__sorted",
          "relName" : "__sort11110"
        }
      } ],
      "input" : {
        "operator" : "sort",
        "gpu" : false,
        "rowType" : [ {
          "relName" : "__sort11110",
          "attrName" : "lo_revenue"
        }, {
          "relName" : "__sort11110",
          "attrName" : "d_year"
        }, {
          "relName" : "__sort11110",
          "attrName" : "p_brand1"
        } ],
        "e" : [ {
          "direction" : "ASC",
          "expression" : {
            "expression" : "recordProjection",
            "e" : {
              "expression" : "argument",
              "attributes" : [ {
                "attrName" : "d_year",
                "relName" : "subsetPelagoProject#11109"
              } ],
              "type" : {
                "relName" : "subsetPelagoProject#11109",
                "type" : "record"
              },
              "argNo" : -1
            },
            "attribute" : {
              "attrName" : "d_year",
              "relName" : "subsetPelagoProject#11109"
            },
            "register_as" : {
              "attrName" : "d_year",
              "relName" : "__sort11110"
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
                "relName" : "subsetPelagoProject#11109"
              } ],
              "type" : {
                "relName" : "subsetPelagoProject#11109",
                "type" : "record"
              },
              "argNo" : -1
            },
            "attribute" : {
              "attrName" : "p_brand1",
              "relName" : "subsetPelagoProject#11109"
            },
            "register_as" : {
              "attrName" : "p_brand1",
              "relName" : "__sort11110"
            }
          }
        }, {
          "direction" : "NONE",
          "expression" : {
            "expression" : "recordProjection",
            "e" : {
              "expression" : "argument",
              "attributes" : [ {
                "attrName" : "lo_revenue",
                "relName" : "subsetPelagoProject#11109"
              } ],
              "type" : {
                "relName" : "subsetPelagoProject#11109",
                "type" : "record"
              },
              "argNo" : -1
            },
            "attribute" : {
              "attrName" : "lo_revenue",
              "relName" : "subsetPelagoProject#11109"
            },
            "register_as" : {
              "attrName" : "lo_revenue",
              "relName" : "__sort11110"
            }
          }
        } ],
        "granularity" : "thread",
        "input" : {
          "operator" : "project",
          "gpu" : false,
          "relName" : "subsetPelagoProject#11109",
          "e" : [ {
            "expression" : "recordProjection",
            "e" : {
              "expression" : "argument",
              "attributes" : [ {
                "attrName" : "lo_revenue",
                "relName" : "subsetPelagoProject#11109"
              } ],
              "type" : {
                "relName" : "subsetPelagoProject#11109",
                "type" : "record"
              },
              "argNo" : -1
            },
            "attribute" : {
              "attrName" : "lo_revenue",
              "relName" : "subsetPelagoProject#11109"
            },
            "register_as" : {
              "attrName" : "lo_revenue",
              "relName" : "subsetPelagoProject#11109"
            }
          }, {
            "expression" : "recordProjection",
            "e" : {
              "expression" : "argument",
              "attributes" : [ {
                "attrName" : "d_year",
                "relName" : "subsetPelagoProject#11109"
              } ],
              "type" : {
                "relName" : "subsetPelagoProject#11109",
                "type" : "record"
              },
              "argNo" : -1
            },
            "attribute" : {
              "attrName" : "d_year",
              "relName" : "subsetPelagoProject#11109"
            },
            "register_as" : {
              "attrName" : "d_year",
              "relName" : "subsetPelagoProject#11109"
            }
          }, {
            "expression" : "recordProjection",
            "e" : {
              "expression" : "argument",
              "attributes" : [ {
                "attrName" : "p_brand1",
                "relName" : "subsetPelagoProject#11109"
              } ],
              "type" : {
                "relName" : "subsetPelagoProject#11109",
                "type" : "record"
              },
              "argNo" : -1
            },
            "attribute" : {
              "attrName" : "p_brand1",
              "relName" : "subsetPelagoProject#11109"
            },
            "register_as" : {
              "attrName" : "p_brand1",
              "relName" : "subsetPelagoProject#11109"
            }
          } ],
          "input" : {
            "operator" : "unpack",
            "gpu" : false,
            "projections" : [ {
              "expression" : "recordProjection",
              "e" : {
                "expression" : "argument",
                "attributes" : [ {
                  "attrName" : "d_year",
                  "relName" : "subsetPelagoProject#11109"
                } ],
                "type" : {
                  "relName" : "subsetPelagoProject#11109",
                  "type" : "record"
                },
                "argNo" : -1
              },
              "attribute" : {
                "attrName" : "d_year",
                "relName" : "subsetPelagoProject#11109"
              }
            }, {
              "expression" : "recordProjection",
              "e" : {
                "expression" : "argument",
                "attributes" : [ {
                  "attrName" : "p_brand1",
                  "relName" : "subsetPelagoProject#11109"
                } ],
                "type" : {
                  "relName" : "subsetPelagoProject#11109",
                  "type" : "record"
                },
                "argNo" : -1
              },
              "attribute" : {
                "attrName" : "p_brand1",
                "relName" : "subsetPelagoProject#11109"
              }
            }, {
              "expression" : "recordProjection",
              "e" : {
                "expression" : "argument",
                "attributes" : [ {
                  "attrName" : "lo_revenue",
                  "relName" : "subsetPelagoProject#11109"
                } ],
                "type" : {
                  "relName" : "subsetPelagoProject#11109",
                  "type" : "record"
                },
                "argNo" : -1
              },
              "attribute" : {
                "attrName" : "lo_revenue",
                "relName" : "subsetPelagoProject#11109"
              }
            } ],
            "input" : {
              "operator" : "mem-move-device",
              "projections" : [ {
                "relName" : "subsetPelagoProject#11109",
                "attrName" : "d_year",
                "isBlock" : true
              }, {
                "relName" : "subsetPelagoProject#11109",
                "attrName" : "p_brand1",
                "isBlock" : true
              }, {
                "relName" : "subsetPelagoProject#11109",
                "attrName" : "lo_revenue",
                "isBlock" : true
              } ],
              "input" : {
                "operator" : "gpu-to-cpu",
                "projections" : [ {
                  "relName" : "subsetPelagoProject#11109",
                  "attrName" : "d_year",
                  "isBlock" : true
                }, {
                  "relName" : "subsetPelagoProject#11109",
                  "attrName" : "p_brand1",
                  "isBlock" : true
                }, {
                  "relName" : "subsetPelagoProject#11109",
                  "attrName" : "lo_revenue",
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
                        "relName" : "subsetPelagoProject#11109"
                      } ],
                      "type" : {
                        "relName" : "subsetPelagoProject#11109",
                        "type" : "record"
                      },
                      "argNo" : -1
                    },
                    "attribute" : {
                      "attrName" : "d_year",
                      "relName" : "subsetPelagoProject#11109"
                    }
                  }, {
                    "expression" : "recordProjection",
                    "e" : {
                      "expression" : "argument",
                      "attributes" : [ {
                        "attrName" : "p_brand1",
                        "relName" : "subsetPelagoProject#11109"
                      } ],
                      "type" : {
                        "relName" : "subsetPelagoProject#11109",
                        "type" : "record"
                      },
                      "argNo" : -1
                    },
                    "attribute" : {
                      "attrName" : "p_brand1",
                      "relName" : "subsetPelagoProject#11109"
                    }
                  }, {
                    "expression" : "recordProjection",
                    "e" : {
                      "expression" : "argument",
                      "attributes" : [ {
                        "attrName" : "lo_revenue",
                        "relName" : "subsetPelagoProject#11109"
                      } ],
                      "type" : {
                        "relName" : "subsetPelagoProject#11109",
                        "type" : "record"
                      },
                      "argNo" : -1
                    },
                    "attribute" : {
                      "attrName" : "lo_revenue",
                      "relName" : "subsetPelagoProject#11109"
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
                          "relName" : "subsetPelagoUnpack#11104"
                        } ],
                        "type" : {
                          "relName" : "subsetPelagoUnpack#11104",
                          "type" : "record"
                        },
                        "argNo" : -1
                      },
                      "attribute" : {
                        "attrName" : "d_year",
                        "relName" : "subsetPelagoUnpack#11104"
                      },
                      "register_as" : {
                        "attrName" : "d_year",
                        "relName" : "subsetPelagoProject#11109"
                      }
                    }, {
                      "expression" : "recordProjection",
                      "e" : {
                        "expression" : "argument",
                        "attributes" : [ {
                          "attrName" : "p_brand1",
                          "relName" : "subsetPelagoUnpack#11104"
                        } ],
                        "type" : {
                          "relName" : "subsetPelagoUnpack#11104",
                          "type" : "record"
                        },
                        "argNo" : -1
                      },
                      "attribute" : {
                        "attrName" : "p_brand1",
                        "relName" : "subsetPelagoUnpack#11104"
                      },
                      "register_as" : {
                        "attrName" : "p_brand1",
                        "relName" : "subsetPelagoProject#11109"
                      }
                    } ],
                    "e" : [ {
                      "m" : "sum",
                      "e" : {
                        "expression" : "recordProjection",
                        "e" : {
                          "expression" : "argument",
                          "attributes" : [ {
                            "attrName" : "lo_revenue",
                            "relName" : "subsetPelagoUnpack#11104"
                          } ],
                          "type" : {
                            "relName" : "subsetPelagoUnpack#11104",
                            "type" : "record"
                          },
                          "argNo" : -1
                        },
                        "attribute" : {
                          "attrName" : "lo_revenue",
                          "relName" : "subsetPelagoUnpack#11104"
                        },
                        "register_as" : {
                          "attrName" : "lo_revenue",
                          "relName" : "subsetPelagoProject#11109"
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
                            "relName" : "subsetPelagoUnpack#11104"
                          } ],
                          "type" : {
                            "relName" : "subsetPelagoUnpack#11104",
                            "type" : "record"
                          },
                          "argNo" : -1
                        },
                        "attribute" : {
                          "attrName" : "d_year",
                          "relName" : "subsetPelagoUnpack#11104"
                        }
                      }, {
                        "expression" : "recordProjection",
                        "e" : {
                          "expression" : "argument",
                          "attributes" : [ {
                            "attrName" : "p_brand1",
                            "relName" : "subsetPelagoUnpack#11104"
                          } ],
                          "type" : {
                            "relName" : "subsetPelagoUnpack#11104",
                            "type" : "record"
                          },
                          "argNo" : -1
                        },
                        "attribute" : {
                          "attrName" : "p_brand1",
                          "relName" : "subsetPelagoUnpack#11104"
                        }
                      }, {
                        "expression" : "recordProjection",
                        "e" : {
                          "expression" : "argument",
                          "attributes" : [ {
                            "attrName" : "lo_revenue",
                            "relName" : "subsetPelagoUnpack#11104"
                          } ],
                          "type" : {
                            "relName" : "subsetPelagoUnpack#11104",
                            "type" : "record"
                          },
                          "argNo" : -1
                        },
                        "attribute" : {
                          "attrName" : "lo_revenue",
                          "relName" : "subsetPelagoUnpack#11104"
                        }
                      } ],
                      "input" : {
                        "operator" : "cpu-to-gpu",
                        "projections" : [ {
                          "relName" : "subsetPelagoUnpack#11104",
                          "attrName" : "d_year",
                          "isBlock" : true
                        }, {
                          "relName" : "subsetPelagoUnpack#11104",
                          "attrName" : "p_brand1",
                          "isBlock" : true
                        }, {
                          "relName" : "subsetPelagoUnpack#11104",
                          "attrName" : "lo_revenue",
                          "isBlock" : true
                        } ],
                        "queueSize" : 262144,
                        "granularity" : "thread",
                        "input" : {
                          "operator" : "mem-move-device",
                          "projections" : [ {
                            "relName" : "subsetPelagoUnpack#11104",
                            "attrName" : "d_year",
                            "isBlock" : true
                          }, {
                            "relName" : "subsetPelagoUnpack#11104",
                            "attrName" : "p_brand1",
                            "isBlock" : true
                          }, {
                            "relName" : "subsetPelagoUnpack#11104",
                            "attrName" : "lo_revenue",
                            "isBlock" : true
                          } ],
                          "input" : {
                            "operator" : "router",
                            "gpu" : false,
                            "projections" : [ {
                              "relName" : "subsetPelagoUnpack#11104",
                              "attrName" : "d_year",
                              "isBlock" : true
                            }, {
                              "relName" : "subsetPelagoUnpack#11104",
                              "attrName" : "p_brand1",
                              "isBlock" : true
                            }, {
                              "relName" : "subsetPelagoUnpack#11104",
                              "attrName" : "lo_revenue",
                              "isBlock" : true
                            } ],
                            "numOfParents" : 1,
                            "producers" : 2,
                            "slack" : 8,
                            "cpu_targets" : false,
                            "input" : {
                              "operator" : "mem-move-device",
                              "projections" : [ {
                                "relName" : "subsetPelagoUnpack#11104",
                                "attrName" : "d_year",
                                "isBlock" : true
                              }, {
                                "relName" : "subsetPelagoUnpack#11104",
                                "attrName" : "p_brand1",
                                "isBlock" : true
                              }, {
                                "relName" : "subsetPelagoUnpack#11104",
                                "attrName" : "lo_revenue",
                                "isBlock" : true
                              } ],
                              "input" : {
                                "operator" : "gpu-to-cpu",
                                "projections" : [ {
                                  "relName" : "subsetPelagoUnpack#11104",
                                  "attrName" : "d_year",
                                  "isBlock" : true
                                }, {
                                  "relName" : "subsetPelagoUnpack#11104",
                                  "attrName" : "p_brand1",
                                  "isBlock" : true
                                }, {
                                  "relName" : "subsetPelagoUnpack#11104",
                                  "attrName" : "lo_revenue",
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
                                        "relName" : "subsetPelagoUnpack#11104"
                                      } ],
                                      "type" : {
                                        "relName" : "subsetPelagoUnpack#11104",
                                        "type" : "record"
                                      },
                                      "argNo" : -1
                                    },
                                    "attribute" : {
                                      "attrName" : "d_year",
                                      "relName" : "subsetPelagoUnpack#11104"
                                    }
                                  }, {
                                    "expression" : "recordProjection",
                                    "e" : {
                                      "expression" : "argument",
                                      "attributes" : [ {
                                        "attrName" : "p_brand1",
                                        "relName" : "subsetPelagoUnpack#11104"
                                      } ],
                                      "type" : {
                                        "relName" : "subsetPelagoUnpack#11104",
                                        "type" : "record"
                                      },
                                      "argNo" : -1
                                    },
                                    "attribute" : {
                                      "attrName" : "p_brand1",
                                      "relName" : "subsetPelagoUnpack#11104"
                                    }
                                  }, {
                                    "expression" : "recordProjection",
                                    "e" : {
                                      "expression" : "argument",
                                      "attributes" : [ {
                                        "attrName" : "lo_revenue",
                                        "relName" : "subsetPelagoUnpack#11104"
                                      } ],
                                      "type" : {
                                        "relName" : "subsetPelagoUnpack#11104",
                                        "type" : "record"
                                      },
                                      "argNo" : -1
                                    },
                                    "attribute" : {
                                      "attrName" : "lo_revenue",
                                      "relName" : "subsetPelagoUnpack#11104"
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
                                          "relName" : "subsetPelagoProject#11098"
                                        } ],
                                        "type" : {
                                          "relName" : "subsetPelagoProject#11098",
                                          "type" : "record"
                                        },
                                        "argNo" : -1
                                      },
                                      "attribute" : {
                                        "attrName" : "d_year",
                                        "relName" : "subsetPelagoProject#11098"
                                      },
                                      "register_as" : {
                                        "attrName" : "d_year",
                                        "relName" : "subsetPelagoUnpack#11104"
                                      }
                                    }, {
                                      "expression" : "recordProjection",
                                      "e" : {
                                        "expression" : "argument",
                                        "attributes" : [ {
                                          "attrName" : "p_brand1",
                                          "relName" : "subsetPelagoProject#11098"
                                        } ],
                                        "type" : {
                                          "relName" : "subsetPelagoProject#11098",
                                          "type" : "record"
                                        },
                                        "argNo" : -1
                                      },
                                      "attribute" : {
                                        "attrName" : "p_brand1",
                                        "relName" : "subsetPelagoProject#11098"
                                      },
                                      "register_as" : {
                                        "attrName" : "p_brand1",
                                        "relName" : "subsetPelagoUnpack#11104"
                                      }
                                    } ],
                                    "e" : [ {
                                      "m" : "sum",
                                      "e" : {
                                        "expression" : "recordProjection",
                                        "e" : {
                                          "expression" : "argument",
                                          "attributes" : [ {
                                            "attrName" : "lo_revenue",
                                            "relName" : "subsetPelagoProject#11098"
                                          } ],
                                          "type" : {
                                            "relName" : "subsetPelagoProject#11098",
                                            "type" : "record"
                                          },
                                          "argNo" : -1
                                        },
                                        "attribute" : {
                                          "attrName" : "lo_revenue",
                                          "relName" : "subsetPelagoProject#11098"
                                        },
                                        "register_as" : {
                                          "attrName" : "lo_revenue",
                                          "relName" : "subsetPelagoUnpack#11104"
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
                                      "relName" : "subsetPelagoProject#11098",
                                      "e" : [ {
                                        "expression" : "recordProjection",
                                        "e" : {
                                          "expression" : "argument",
                                          "attributes" : [ {
                                            "attrName" : "d_year",
                                            "relName" : "subsetPelagoProject#11098"
                                          } ],
                                          "type" : {
                                            "relName" : "subsetPelagoProject#11098",
                                            "type" : "record"
                                          },
                                          "argNo" : -1
                                        },
                                        "attribute" : {
                                          "attrName" : "d_year",
                                          "relName" : "subsetPelagoProject#11098"
                                        },
                                        "register_as" : {
                                          "attrName" : "d_year",
                                          "relName" : "subsetPelagoProject#11098"
                                        }
                                      }, {
                                        "expression" : "recordProjection",
                                        "e" : {
                                          "expression" : "argument",
                                          "attributes" : [ {
                                            "attrName" : "p_brand1",
                                            "relName" : "subsetPelagoProject#11098"
                                          } ],
                                          "type" : {
                                            "relName" : "subsetPelagoProject#11098",
                                            "type" : "record"
                                          },
                                          "argNo" : -1
                                        },
                                        "attribute" : {
                                          "attrName" : "p_brand1",
                                          "relName" : "subsetPelagoProject#11098"
                                        },
                                        "register_as" : {
                                          "attrName" : "p_brand1",
                                          "relName" : "subsetPelagoProject#11098"
                                        }
                                      }, {
                                        "expression" : "recordProjection",
                                        "e" : {
                                          "expression" : "argument",
                                          "attributes" : [ {
                                            "attrName" : "lo_revenue",
                                            "relName" : "subsetPelagoProject#11098"
                                          } ],
                                          "type" : {
                                            "relName" : "subsetPelagoProject#11098",
                                            "type" : "record"
                                          },
                                          "argNo" : -1
                                        },
                                        "attribute" : {
                                          "attrName" : "lo_revenue",
                                          "relName" : "subsetPelagoProject#11098"
                                        },
                                        "register_as" : {
                                          "attrName" : "lo_revenue",
                                          "relName" : "subsetPelagoProject#11098"
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
                                            "attrName" : "$2",
                                            "relName" : "subsetPelagoProject#11098"
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
                                              "relName" : "subsetPelagoProject#11098"
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
                                              "relName" : "subsetPelagoProject#11098"
                                            }
                                          },
                                          "packet" : 2,
                                          "offset" : 0
                                        } ],
                                        "build_w" : [ 64, 32, 32 ],
                                        "build_input" : {
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
                                        },
                                        "probe_k" : {
                                          "expression" : "recordProjection",
                                          "e" : {
                                            "expression" : "argument",
                                            "attributes" : [ {
                                              "attrName" : "lo_orderdate",
                                              "relName" : "subsetPelagoProject#11096"
                                            } ],
                                            "type" : {
                                              "relName" : "subsetPelagoProject#11096",
                                              "type" : "record"
                                            },
                                            "argNo" : -1
                                          },
                                          "attribute" : {
                                            "attrName" : "lo_orderdate",
                                            "relName" : "subsetPelagoProject#11096"
                                          },
                                          "register_as" : {
                                            "attrName" : "$0",
                                            "relName" : "subsetPelagoProject#11098"
                                          }
                                        },
                                        "probe_e" : [ {
                                          "e" : {
                                            "expression" : "recordProjection",
                                            "e" : {
                                              "expression" : "argument",
                                              "attributes" : [ {
                                                "attrName" : "lo_orderdate",
                                                "relName" : "subsetPelagoProject#11096"
                                              } ],
                                              "type" : {
                                                "relName" : "subsetPelagoProject#11096",
                                                "type" : "record"
                                              },
                                              "argNo" : -1
                                            },
                                            "attribute" : {
                                              "attrName" : "lo_orderdate",
                                              "relName" : "subsetPelagoProject#11096"
                                            },
                                            "register_as" : {
                                              "attrName" : "lo_orderdate",
                                              "relName" : "subsetPelagoProject#11098"
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
                                                "attrName" : "lo_revenue",
                                                "relName" : "subsetPelagoProject#11096"
                                              } ],
                                              "type" : {
                                                "relName" : "subsetPelagoProject#11096",
                                                "type" : "record"
                                              },
                                              "argNo" : -1
                                            },
                                            "attribute" : {
                                              "attrName" : "lo_revenue",
                                              "relName" : "subsetPelagoProject#11096"
                                            },
                                            "register_as" : {
                                              "attrName" : "lo_revenue",
                                              "relName" : "subsetPelagoProject#11098"
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
                                                "attrName" : "p_brand1",
                                                "relName" : "subsetPelagoProject#11096"
                                              } ],
                                              "type" : {
                                                "relName" : "subsetPelagoProject#11096",
                                                "type" : "record"
                                              },
                                              "argNo" : -1
                                            },
                                            "attribute" : {
                                              "attrName" : "p_brand1",
                                              "relName" : "subsetPelagoProject#11096"
                                            },
                                            "register_as" : {
                                              "attrName" : "p_brand1",
                                              "relName" : "subsetPelagoProject#11098"
                                            }
                                          },
                                          "packet" : 3,
                                          "offset" : 0
                                        } ],
                                        "probe_w" : [ 64, 32, 32, 32 ],
                                        "hash_bits" : 24,
                                        "maxBuildInputSize" : 2556,
                                        "probe_input" : {
                                          "operator" : "project",
                                          "gpu" : true,
                                          "relName" : "subsetPelagoProject#11096",
                                          "e" : [ {
                                            "expression" : "recordProjection",
                                            "e" : {
                                              "expression" : "argument",
                                              "attributes" : [ {
                                                "attrName" : "lo_orderdate",
                                                "relName" : "subsetPelagoProject#11096"
                                              } ],
                                              "type" : {
                                                "relName" : "subsetPelagoProject#11096",
                                                "type" : "record"
                                              },
                                              "argNo" : -1
                                            },
                                            "attribute" : {
                                              "attrName" : "lo_orderdate",
                                              "relName" : "subsetPelagoProject#11096"
                                            },
                                            "register_as" : {
                                              "attrName" : "lo_orderdate",
                                              "relName" : "subsetPelagoProject#11096"
                                            }
                                          }, {
                                            "expression" : "recordProjection",
                                            "e" : {
                                              "expression" : "argument",
                                              "attributes" : [ {
                                                "attrName" : "lo_revenue",
                                                "relName" : "subsetPelagoProject#11096"
                                              } ],
                                              "type" : {
                                                "relName" : "subsetPelagoProject#11096",
                                                "type" : "record"
                                              },
                                              "argNo" : -1
                                            },
                                            "attribute" : {
                                              "attrName" : "lo_revenue",
                                              "relName" : "subsetPelagoProject#11096"
                                            },
                                            "register_as" : {
                                              "attrName" : "lo_revenue",
                                              "relName" : "subsetPelagoProject#11096"
                                            }
                                          }, {
                                            "expression" : "recordProjection",
                                            "e" : {
                                              "expression" : "argument",
                                              "attributes" : [ {
                                                "attrName" : "p_brand1",
                                                "relName" : "subsetPelagoProject#11096"
                                              } ],
                                              "type" : {
                                                "relName" : "subsetPelagoProject#11096",
                                                "type" : "record"
                                              },
                                              "argNo" : -1
                                            },
                                            "attribute" : {
                                              "attrName" : "p_brand1",
                                              "relName" : "subsetPelagoProject#11096"
                                            },
                                            "register_as" : {
                                              "attrName" : "p_brand1",
                                              "relName" : "subsetPelagoProject#11096"
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
                                                  "relName" : "subsetPelagoProject#11085"
                                                } ],
                                                "type" : {
                                                  "relName" : "subsetPelagoProject#11085",
                                                  "type" : "record"
                                                },
                                                "argNo" : -1
                                              },
                                              "attribute" : {
                                                "attrName" : "s_suppkey",
                                                "relName" : "subsetPelagoProject#11085"
                                              },
                                              "register_as" : {
                                                "attrName" : "$1",
                                                "relName" : "subsetPelagoProject#11096"
                                              }
                                            },
                                            "build_e" : [ ],
                                            "build_w" : [ 64 ],
                                            "build_input" : {
                                              "operator" : "project",
                                              "gpu" : true,
                                              "relName" : "subsetPelagoProject#11085",
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
                                                  "relName" : "subsetPelagoProject#11085"
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
                                                        "attrName" : "s_region",
                                                        "relName" : "inputs/ssbm1000/supplier.csv"
                                                      } ],
                                                      "type" : {
                                                        "relName" : "inputs/ssbm1000/supplier.csv",
                                                        "type" : "record"
                                                      },
                                                      "argNo" : -1
                                                    },
                                                    "attribute" : {
                                                      "attrName" : "s_region",
                                                      "relName" : "inputs/ssbm1000/supplier.csv"
                                                    }
                                                  },
                                                  "right" : {
                                                    "expression" : "dstring",
                                                    "v" : "ASIA",
                                                    "dict" : {
                                                      "path" : "inputs/ssbm1000/supplier.csv.s_region.dict"
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
                                                        "attrName" : "s_region",
                                                        "relName" : "inputs/ssbm1000/supplier.csv"
                                                      } ],
                                                      "type" : {
                                                        "relName" : "inputs/ssbm1000/supplier.csv",
                                                        "type" : "record"
                                                      },
                                                      "argNo" : -1
                                                    },
                                                    "attribute" : {
                                                      "attrName" : "s_region",
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
                                                      "attrName" : "s_region",
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
                                                        "attrName" : "s_region",
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
                                                          "attrName" : "s_region",
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
                                                              "attrName" : "s_region"
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
                                                  "relName" : "subsetPelagoProject#11094"
                                                } ],
                                                "type" : {
                                                  "relName" : "subsetPelagoProject#11094",
                                                  "type" : "record"
                                                },
                                                "argNo" : -1
                                              },
                                              "attribute" : {
                                                "attrName" : "lo_suppkey",
                                                "relName" : "subsetPelagoProject#11094"
                                              },
                                              "register_as" : {
                                                "attrName" : "$0",
                                                "relName" : "subsetPelagoProject#11096"
                                              }
                                            },
                                            "probe_e" : [ {
                                              "e" : {
                                                "expression" : "recordProjection",
                                                "e" : {
                                                  "expression" : "argument",
                                                  "attributes" : [ {
                                                    "attrName" : "lo_suppkey",
                                                    "relName" : "subsetPelagoProject#11094"
                                                  } ],
                                                  "type" : {
                                                    "relName" : "subsetPelagoProject#11094",
                                                    "type" : "record"
                                                  },
                                                  "argNo" : -1
                                                },
                                                "attribute" : {
                                                  "attrName" : "lo_suppkey",
                                                  "relName" : "subsetPelagoProject#11094"
                                                },
                                                "register_as" : {
                                                  "attrName" : "lo_suppkey",
                                                  "relName" : "subsetPelagoProject#11096"
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
                                                    "attrName" : "lo_orderdate",
                                                    "relName" : "subsetPelagoProject#11094"
                                                  } ],
                                                  "type" : {
                                                    "relName" : "subsetPelagoProject#11094",
                                                    "type" : "record"
                                                  },
                                                  "argNo" : -1
                                                },
                                                "attribute" : {
                                                  "attrName" : "lo_orderdate",
                                                  "relName" : "subsetPelagoProject#11094"
                                                },
                                                "register_as" : {
                                                  "attrName" : "lo_orderdate",
                                                  "relName" : "subsetPelagoProject#11096"
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
                                                    "attrName" : "lo_revenue",
                                                    "relName" : "subsetPelagoProject#11094"
                                                  } ],
                                                  "type" : {
                                                    "relName" : "subsetPelagoProject#11094",
                                                    "type" : "record"
                                                  },
                                                  "argNo" : -1
                                                },
                                                "attribute" : {
                                                  "attrName" : "lo_revenue",
                                                  "relName" : "subsetPelagoProject#11094"
                                                },
                                                "register_as" : {
                                                  "attrName" : "lo_revenue",
                                                  "relName" : "subsetPelagoProject#11096"
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
                                                    "relName" : "subsetPelagoProject#11094"
                                                  } ],
                                                  "type" : {
                                                    "relName" : "subsetPelagoProject#11094",
                                                    "type" : "record"
                                                  },
                                                  "argNo" : -1
                                                },
                                                "attribute" : {
                                                  "attrName" : "p_brand1",
                                                  "relName" : "subsetPelagoProject#11094"
                                                },
                                                "register_as" : {
                                                  "attrName" : "p_brand1",
                                                  "relName" : "subsetPelagoProject#11096"
                                                }
                                              },
                                              "packet" : 4,
                                              "offset" : 0
                                            } ],
                                            "probe_w" : [ 64, 32, 32, 32, 32 ],
                                            "hash_bits" : 20,
                                            "maxBuildInputSize" : 2000000,
                                            "probe_input" : {
                                              "operator" : "project",
                                              "gpu" : true,
                                              "relName" : "subsetPelagoProject#11094",
                                              "e" : [ {
                                                "expression" : "recordProjection",
                                                "e" : {
                                                  "expression" : "argument",
                                                  "attributes" : [ {
                                                    "attrName" : "lo_suppkey",
                                                    "relName" : "subsetPelagoProject#11094"
                                                  } ],
                                                  "type" : {
                                                    "relName" : "subsetPelagoProject#11094",
                                                    "type" : "record"
                                                  },
                                                  "argNo" : -1
                                                },
                                                "attribute" : {
                                                  "attrName" : "lo_suppkey",
                                                  "relName" : "subsetPelagoProject#11094"
                                                },
                                                "register_as" : {
                                                  "attrName" : "lo_suppkey",
                                                  "relName" : "subsetPelagoProject#11094"
                                                }
                                              }, {
                                                "expression" : "recordProjection",
                                                "e" : {
                                                  "expression" : "argument",
                                                  "attributes" : [ {
                                                    "attrName" : "lo_orderdate",
                                                    "relName" : "subsetPelagoProject#11094"
                                                  } ],
                                                  "type" : {
                                                    "relName" : "subsetPelagoProject#11094",
                                                    "type" : "record"
                                                  },
                                                  "argNo" : -1
                                                },
                                                "attribute" : {
                                                  "attrName" : "lo_orderdate",
                                                  "relName" : "subsetPelagoProject#11094"
                                                },
                                                "register_as" : {
                                                  "attrName" : "lo_orderdate",
                                                  "relName" : "subsetPelagoProject#11094"
                                                }
                                              }, {
                                                "expression" : "recordProjection",
                                                "e" : {
                                                  "expression" : "argument",
                                                  "attributes" : [ {
                                                    "attrName" : "lo_revenue",
                                                    "relName" : "subsetPelagoProject#11094"
                                                  } ],
                                                  "type" : {
                                                    "relName" : "subsetPelagoProject#11094",
                                                    "type" : "record"
                                                  },
                                                  "argNo" : -1
                                                },
                                                "attribute" : {
                                                  "attrName" : "lo_revenue",
                                                  "relName" : "subsetPelagoProject#11094"
                                                },
                                                "register_as" : {
                                                  "attrName" : "lo_revenue",
                                                  "relName" : "subsetPelagoProject#11094"
                                                }
                                              }, {
                                                "expression" : "recordProjection",
                                                "e" : {
                                                  "expression" : "argument",
                                                  "attributes" : [ {
                                                    "attrName" : "p_brand1",
                                                    "relName" : "subsetPelagoProject#11094"
                                                  } ],
                                                  "type" : {
                                                    "relName" : "subsetPelagoProject#11094",
                                                    "type" : "record"
                                                  },
                                                  "argNo" : -1
                                                },
                                                "attribute" : {
                                                  "attrName" : "p_brand1",
                                                  "relName" : "subsetPelagoProject#11094"
                                                },
                                                "register_as" : {
                                                  "attrName" : "p_brand1",
                                                  "relName" : "subsetPelagoProject#11094"
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
                                                    "attrName" : "$2",
                                                    "relName" : "subsetPelagoProject#11094"
                                                  }
                                                },
                                                "build_e" : [ {
                                                  "e" : {
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
                                                      "relName" : "subsetPelagoProject#11094"
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
                                                      "relName" : "subsetPelagoProject#11094"
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
                                                    "expression" : "and",
                                                    "left" : {
                                                      "expression" : "ge",
                                                      "left" : {
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
                                                      },
                                                      "right" : {
                                                        "expression" : "dstring",
                                                        "v" : "MFGR#2221",
                                                        "dict" : {
                                                          "path" : "inputs/ssbm1000/part.csv.p_brand1.dict"
                                                        }
                                                      }
                                                    },
                                                    "right" : {
                                                      "expression" : "le",
                                                      "left" : {
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
                                                      },
                                                      "right" : {
                                                        "expression" : "dstring",
                                                        "v" : "MFGR#2228",
                                                        "dict" : {
                                                          "path" : "inputs/ssbm1000/part.csv.p_brand1.dict"
                                                        }
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
                                                },
                                                "probe_k" : {
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
                                                    "attrName" : "$0",
                                                    "relName" : "subsetPelagoProject#11094"
                                                  }
                                                },
                                                "probe_e" : [ {
                                                  "e" : {
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
                                                      "relName" : "subsetPelagoProject#11094"
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
                                                      "relName" : "subsetPelagoProject#11094"
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
                                                      "relName" : "subsetPelagoProject#11094"
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
                                                    },
                                                    "register_as" : {
                                                      "attrName" : "lo_revenue",
                                                      "relName" : "subsetPelagoProject#11094"
                                                    }
                                                  },
                                                  "packet" : 4,
                                                  "offset" : 0
                                                } ],
                                                "probe_w" : [ 64, 32, 32, 32, 32 ],
                                                "hash_bits" : 16,
                                                "maxBuildInputSize" : 320000,
                                                "probe_input" : {
                                                  "operator" : "unpack",
                                                  "gpu" : true,
                                                  "projections" : [ {
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
                                                  } ],
                                                  "input" : {
                                                    "operator" : "cpu-to-gpu",
                                                    "projections" : [ {
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
                                                    } ],
                                                    "queueSize" : 262144,
                                                    "granularity" : "thread",
                                                    "input" : {
                                                      "operator" : "mem-move-device",
                                                      "projections" : [ {
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
                                                      } ],
                                                      "do_transfer": [
                                                        false,
                                                        false,
                                                        false,
                                                        false
                                                      ],
                                                      "input" : {
                                                        "operator" : "router",
                                                        "gpu" : false,
                                                        "projections" : [ {
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
}
