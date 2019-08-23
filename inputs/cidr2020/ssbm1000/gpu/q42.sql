{
  "operator" : "print",
  "gpu" : false,
  "e" : [ {
    "expression" : "recordProjection",
    "e" : {
      "expression" : "argument",
      "attributes" : [ {
        "attrName" : "d_year",
        "relName" : "subsetPelagoSort#11600"
      } ],
      "type" : {
        "relName" : "subsetPelagoSort#11600",
        "type" : "record"
      },
      "argNo" : -1
    },
    "attribute" : {
      "attrName" : "d_year",
      "relName" : "subsetPelagoSort#11600"
    },
    "register_as" : {
      "attrName" : "d_year",
      "relName" : "print11601"
    }
  }, {
    "expression" : "recordProjection",
    "e" : {
      "expression" : "argument",
      "attributes" : [ {
        "attrName" : "s_nation",
        "relName" : "subsetPelagoSort#11600"
      } ],
      "type" : {
        "relName" : "subsetPelagoSort#11600",
        "type" : "record"
      },
      "argNo" : -1
    },
    "attribute" : {
      "attrName" : "s_nation",
      "relName" : "subsetPelagoSort#11600"
    },
    "register_as" : {
      "attrName" : "s_nation",
      "relName" : "print11601"
    }
  }, {
    "expression" : "recordProjection",
    "e" : {
      "expression" : "argument",
      "attributes" : [ {
        "attrName" : "p_category",
        "relName" : "subsetPelagoSort#11600"
      } ],
      "type" : {
        "relName" : "subsetPelagoSort#11600",
        "type" : "record"
      },
      "argNo" : -1
    },
    "attribute" : {
      "attrName" : "p_category",
      "relName" : "subsetPelagoSort#11600"
    },
    "register_as" : {
      "attrName" : "p_category",
      "relName" : "print11601"
    }
  }, {
    "expression" : "recordProjection",
    "e" : {
      "expression" : "argument",
      "attributes" : [ {
        "attrName" : "profit",
        "relName" : "subsetPelagoSort#11600"
      } ],
      "type" : {
        "relName" : "subsetPelagoSort#11600",
        "type" : "record"
      },
      "argNo" : -1
    },
    "attribute" : {
      "attrName" : "profit",
      "relName" : "subsetPelagoSort#11600"
    },
    "register_as" : {
      "attrName" : "profit",
      "relName" : "print11601"
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
            "relName" : "__sort11600"
          } ],
          "expression" : "argument",
          "type" : {
            "type" : "record",
            "relName" : "__sort11600"
          },
          "argNo" : 1
        },
        "expression" : "recordProjection",
        "attribute" : {
          "attrName" : "__sorted",
          "relName" : "__sort11600"
        }
      },
      "expression" : "recordProjection",
      "attribute" : {
        "attrName" : "d_year",
        "relName" : "__sort11600"
      },
      "register_as" : {
        "attrName" : "d_year",
        "relName" : "subsetPelagoSort#11600"
      }
    }, {
      "e" : {
        "e" : {
          "attributes" : [ {
            "attrName" : "__sorted",
            "relName" : "__sort11600"
          } ],
          "expression" : "argument",
          "type" : {
            "type" : "record",
            "relName" : "__sort11600"
          },
          "argNo" : 1
        },
        "expression" : "recordProjection",
        "attribute" : {
          "attrName" : "__sorted",
          "relName" : "__sort11600"
        }
      },
      "expression" : "recordProjection",
      "attribute" : {
        "attrName" : "s_nation",
        "relName" : "__sort11600"
      },
      "register_as" : {
        "attrName" : "s_nation",
        "relName" : "subsetPelagoSort#11600"
      }
    }, {
      "e" : {
        "e" : {
          "attributes" : [ {
            "attrName" : "__sorted",
            "relName" : "__sort11600"
          } ],
          "expression" : "argument",
          "type" : {
            "type" : "record",
            "relName" : "__sort11600"
          },
          "argNo" : 1
        },
        "expression" : "recordProjection",
        "attribute" : {
          "attrName" : "__sorted",
          "relName" : "__sort11600"
        }
      },
      "expression" : "recordProjection",
      "attribute" : {
        "attrName" : "p_category",
        "relName" : "__sort11600"
      },
      "register_as" : {
        "attrName" : "p_category",
        "relName" : "subsetPelagoSort#11600"
      }
    }, {
      "e" : {
        "e" : {
          "attributes" : [ {
            "attrName" : "__sorted",
            "relName" : "__sort11600"
          } ],
          "expression" : "argument",
          "type" : {
            "type" : "record",
            "relName" : "__sort11600"
          },
          "argNo" : 1
        },
        "expression" : "recordProjection",
        "attribute" : {
          "attrName" : "__sorted",
          "relName" : "__sort11600"
        }
      },
      "expression" : "recordProjection",
      "attribute" : {
        "attrName" : "profit",
        "relName" : "__sort11600"
      },
      "register_as" : {
        "attrName" : "profit",
        "relName" : "subsetPelagoSort#11600"
      }
    } ],
    "relName" : "subsetPelagoSort#11600",
    "input" : {
      "operator" : "unpack",
      "gpu" : false,
      "projections" : [ {
        "e" : {
          "attributes" : [ {
            "attrName" : "__sorted",
            "relName" : "__sort11600"
          } ],
          "expression" : "argument",
          "type" : {
            "type" : "record",
            "relName" : "__sort11600"
          },
          "argNo" : 1
        },
        "expression" : "recordProjection",
        "attribute" : {
          "attrName" : "__sorted",
          "relName" : "__sort11600"
        }
      } ],
      "input" : {
        "operator" : "sort",
        "gpu" : false,
        "rowType" : [ {
          "relName" : "__sort11600",
          "attrName" : "d_year"
        }, {
          "relName" : "__sort11600",
          "attrName" : "s_nation"
        }, {
          "relName" : "__sort11600",
          "attrName" : "p_category"
        }, {
          "relName" : "__sort11600",
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
                "relName" : "subsetPelagoUnpack#11599"
              } ],
              "type" : {
                "relName" : "subsetPelagoUnpack#11599",
                "type" : "record"
              },
              "argNo" : -1
            },
            "attribute" : {
              "attrName" : "d_year",
              "relName" : "subsetPelagoUnpack#11599"
            },
            "register_as" : {
              "attrName" : "d_year",
              "relName" : "__sort11600"
            }
          }
        }, {
          "direction" : "ASC",
          "expression" : {
            "expression" : "recordProjection",
            "e" : {
              "expression" : "argument",
              "attributes" : [ {
                "attrName" : "s_nation",
                "relName" : "subsetPelagoUnpack#11599"
              } ],
              "type" : {
                "relName" : "subsetPelagoUnpack#11599",
                "type" : "record"
              },
              "argNo" : -1
            },
            "attribute" : {
              "attrName" : "s_nation",
              "relName" : "subsetPelagoUnpack#11599"
            },
            "register_as" : {
              "attrName" : "s_nation",
              "relName" : "__sort11600"
            }
          }
        }, {
          "direction" : "ASC",
          "expression" : {
            "expression" : "recordProjection",
            "e" : {
              "expression" : "argument",
              "attributes" : [ {
                "attrName" : "p_category",
                "relName" : "subsetPelagoUnpack#11599"
              } ],
              "type" : {
                "relName" : "subsetPelagoUnpack#11599",
                "type" : "record"
              },
              "argNo" : -1
            },
            "attribute" : {
              "attrName" : "p_category",
              "relName" : "subsetPelagoUnpack#11599"
            },
            "register_as" : {
              "attrName" : "p_category",
              "relName" : "__sort11600"
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
                "relName" : "subsetPelagoUnpack#11599"
              } ],
              "type" : {
                "relName" : "subsetPelagoUnpack#11599",
                "type" : "record"
              },
              "argNo" : -1
            },
            "attribute" : {
              "attrName" : "profit",
              "relName" : "subsetPelagoUnpack#11599"
            },
            "register_as" : {
              "attrName" : "profit",
              "relName" : "__sort11600"
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
                "relName" : "subsetPelagoUnpack#11599"
              } ],
              "type" : {
                "relName" : "subsetPelagoUnpack#11599",
                "type" : "record"
              },
              "argNo" : -1
            },
            "attribute" : {
              "attrName" : "d_year",
              "relName" : "subsetPelagoUnpack#11599"
            }
          }, {
            "expression" : "recordProjection",
            "e" : {
              "expression" : "argument",
              "attributes" : [ {
                "attrName" : "s_nation",
                "relName" : "subsetPelagoUnpack#11599"
              } ],
              "type" : {
                "relName" : "subsetPelagoUnpack#11599",
                "type" : "record"
              },
              "argNo" : -1
            },
            "attribute" : {
              "attrName" : "s_nation",
              "relName" : "subsetPelagoUnpack#11599"
            }
          }, {
            "expression" : "recordProjection",
            "e" : {
              "expression" : "argument",
              "attributes" : [ {
                "attrName" : "p_category",
                "relName" : "subsetPelagoUnpack#11599"
              } ],
              "type" : {
                "relName" : "subsetPelagoUnpack#11599",
                "type" : "record"
              },
              "argNo" : -1
            },
            "attribute" : {
              "attrName" : "p_category",
              "relName" : "subsetPelagoUnpack#11599"
            }
          }, {
            "expression" : "recordProjection",
            "e" : {
              "expression" : "argument",
              "attributes" : [ {
                "attrName" : "profit",
                "relName" : "subsetPelagoUnpack#11599"
              } ],
              "type" : {
                "relName" : "subsetPelagoUnpack#11599",
                "type" : "record"
              },
              "argNo" : -1
            },
            "attribute" : {
              "attrName" : "profit",
              "relName" : "subsetPelagoUnpack#11599"
            }
          } ],
          "input" : {
            "operator" : "mem-move-device",
            "projections" : [ {
              "relName" : "subsetPelagoUnpack#11599",
              "attrName" : "d_year",
              "isBlock" : true
            }, {
              "relName" : "subsetPelagoUnpack#11599",
              "attrName" : "s_nation",
              "isBlock" : true
            }, {
              "relName" : "subsetPelagoUnpack#11599",
              "attrName" : "p_category",
              "isBlock" : true
            }, {
              "relName" : "subsetPelagoUnpack#11599",
              "attrName" : "profit",
              "isBlock" : true
            } ],
            "input" : {
              "operator" : "gpu-to-cpu",
              "projections" : [ {
                "relName" : "subsetPelagoUnpack#11599",
                "attrName" : "d_year",
                "isBlock" : true
              }, {
                "relName" : "subsetPelagoUnpack#11599",
                "attrName" : "s_nation",
                "isBlock" : true
              }, {
                "relName" : "subsetPelagoUnpack#11599",
                "attrName" : "p_category",
                "isBlock" : true
              }, {
                "relName" : "subsetPelagoUnpack#11599",
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
                      "relName" : "subsetPelagoUnpack#11599"
                    } ],
                    "type" : {
                      "relName" : "subsetPelagoUnpack#11599",
                      "type" : "record"
                    },
                    "argNo" : -1
                  },
                  "attribute" : {
                    "attrName" : "d_year",
                    "relName" : "subsetPelagoUnpack#11599"
                  }
                }, {
                  "expression" : "recordProjection",
                  "e" : {
                    "expression" : "argument",
                    "attributes" : [ {
                      "attrName" : "s_nation",
                      "relName" : "subsetPelagoUnpack#11599"
                    } ],
                    "type" : {
                      "relName" : "subsetPelagoUnpack#11599",
                      "type" : "record"
                    },
                    "argNo" : -1
                  },
                  "attribute" : {
                    "attrName" : "s_nation",
                    "relName" : "subsetPelagoUnpack#11599"
                  }
                }, {
                  "expression" : "recordProjection",
                  "e" : {
                    "expression" : "argument",
                    "attributes" : [ {
                      "attrName" : "p_category",
                      "relName" : "subsetPelagoUnpack#11599"
                    } ],
                    "type" : {
                      "relName" : "subsetPelagoUnpack#11599",
                      "type" : "record"
                    },
                    "argNo" : -1
                  },
                  "attribute" : {
                    "attrName" : "p_category",
                    "relName" : "subsetPelagoUnpack#11599"
                  }
                }, {
                  "expression" : "recordProjection",
                  "e" : {
                    "expression" : "argument",
                    "attributes" : [ {
                      "attrName" : "profit",
                      "relName" : "subsetPelagoUnpack#11599"
                    } ],
                    "type" : {
                      "relName" : "subsetPelagoUnpack#11599",
                      "type" : "record"
                    },
                    "argNo" : -1
                  },
                  "attribute" : {
                    "attrName" : "profit",
                    "relName" : "subsetPelagoUnpack#11599"
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
                        "relName" : "subsetPelagoUnpack#11595"
                      } ],
                      "type" : {
                        "relName" : "subsetPelagoUnpack#11595",
                        "type" : "record"
                      },
                      "argNo" : -1
                    },
                    "attribute" : {
                      "attrName" : "d_year",
                      "relName" : "subsetPelagoUnpack#11595"
                    },
                    "register_as" : {
                      "attrName" : "d_year",
                      "relName" : "subsetPelagoUnpack#11599"
                    }
                  }, {
                    "expression" : "recordProjection",
                    "e" : {
                      "expression" : "argument",
                      "attributes" : [ {
                        "attrName" : "s_nation",
                        "relName" : "subsetPelagoUnpack#11595"
                      } ],
                      "type" : {
                        "relName" : "subsetPelagoUnpack#11595",
                        "type" : "record"
                      },
                      "argNo" : -1
                    },
                    "attribute" : {
                      "attrName" : "s_nation",
                      "relName" : "subsetPelagoUnpack#11595"
                    },
                    "register_as" : {
                      "attrName" : "s_nation",
                      "relName" : "subsetPelagoUnpack#11599"
                    }
                  }, {
                    "expression" : "recordProjection",
                    "e" : {
                      "expression" : "argument",
                      "attributes" : [ {
                        "attrName" : "p_category",
                        "relName" : "subsetPelagoUnpack#11595"
                      } ],
                      "type" : {
                        "relName" : "subsetPelagoUnpack#11595",
                        "type" : "record"
                      },
                      "argNo" : -1
                    },
                    "attribute" : {
                      "attrName" : "p_category",
                      "relName" : "subsetPelagoUnpack#11595"
                    },
                    "register_as" : {
                      "attrName" : "p_category",
                      "relName" : "subsetPelagoUnpack#11599"
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
                          "relName" : "subsetPelagoUnpack#11595"
                        } ],
                        "type" : {
                          "relName" : "subsetPelagoUnpack#11595",
                          "type" : "record"
                        },
                        "argNo" : -1
                      },
                      "attribute" : {
                        "attrName" : "profit",
                        "relName" : "subsetPelagoUnpack#11595"
                      },
                      "register_as" : {
                        "attrName" : "profit",
                        "relName" : "subsetPelagoUnpack#11599"
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
                          "relName" : "subsetPelagoUnpack#11595"
                        } ],
                        "type" : {
                          "relName" : "subsetPelagoUnpack#11595",
                          "type" : "record"
                        },
                        "argNo" : -1
                      },
                      "attribute" : {
                        "attrName" : "d_year",
                        "relName" : "subsetPelagoUnpack#11595"
                      }
                    }, {
                      "expression" : "recordProjection",
                      "e" : {
                        "expression" : "argument",
                        "attributes" : [ {
                          "attrName" : "s_nation",
                          "relName" : "subsetPelagoUnpack#11595"
                        } ],
                        "type" : {
                          "relName" : "subsetPelagoUnpack#11595",
                          "type" : "record"
                        },
                        "argNo" : -1
                      },
                      "attribute" : {
                        "attrName" : "s_nation",
                        "relName" : "subsetPelagoUnpack#11595"
                      }
                    }, {
                      "expression" : "recordProjection",
                      "e" : {
                        "expression" : "argument",
                        "attributes" : [ {
                          "attrName" : "p_category",
                          "relName" : "subsetPelagoUnpack#11595"
                        } ],
                        "type" : {
                          "relName" : "subsetPelagoUnpack#11595",
                          "type" : "record"
                        },
                        "argNo" : -1
                      },
                      "attribute" : {
                        "attrName" : "p_category",
                        "relName" : "subsetPelagoUnpack#11595"
                      }
                    }, {
                      "expression" : "recordProjection",
                      "e" : {
                        "expression" : "argument",
                        "attributes" : [ {
                          "attrName" : "profit",
                          "relName" : "subsetPelagoUnpack#11595"
                        } ],
                        "type" : {
                          "relName" : "subsetPelagoUnpack#11595",
                          "type" : "record"
                        },
                        "argNo" : -1
                      },
                      "attribute" : {
                        "attrName" : "profit",
                        "relName" : "subsetPelagoUnpack#11595"
                      }
                    } ],
                    "input" : {
                      "operator" : "cpu-to-gpu",
                      "projections" : [ {
                        "relName" : "subsetPelagoUnpack#11595",
                        "attrName" : "d_year",
                        "isBlock" : true
                      }, {
                        "relName" : "subsetPelagoUnpack#11595",
                        "attrName" : "s_nation",
                        "isBlock" : true
                      }, {
                        "relName" : "subsetPelagoUnpack#11595",
                        "attrName" : "p_category",
                        "isBlock" : true
                      }, {
                        "relName" : "subsetPelagoUnpack#11595",
                        "attrName" : "profit",
                        "isBlock" : true
                      } ],
                      "queueSize" : 262144,
                      "granularity" : "thread",
                      "input" : {
                        "operator" : "mem-move-device",
                        "projections" : [ {
                          "relName" : "subsetPelagoUnpack#11595",
                          "attrName" : "d_year",
                          "isBlock" : true
                        }, {
                          "relName" : "subsetPelagoUnpack#11595",
                          "attrName" : "s_nation",
                          "isBlock" : true
                        }, {
                          "relName" : "subsetPelagoUnpack#11595",
                          "attrName" : "p_category",
                          "isBlock" : true
                        }, {
                          "relName" : "subsetPelagoUnpack#11595",
                          "attrName" : "profit",
                          "isBlock" : true
                        } ],
                        "input" : {
                          "operator" : "router",
                          "gpu" : false,
                          "projections" : [ {
                            "relName" : "subsetPelagoUnpack#11595",
                            "attrName" : "d_year",
                            "isBlock" : true
                          }, {
                            "relName" : "subsetPelagoUnpack#11595",
                            "attrName" : "s_nation",
                            "isBlock" : true
                          }, {
                            "relName" : "subsetPelagoUnpack#11595",
                            "attrName" : "p_category",
                            "isBlock" : true
                          }, {
                            "relName" : "subsetPelagoUnpack#11595",
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
                              "relName" : "subsetPelagoUnpack#11595",
                              "attrName" : "d_year",
                              "isBlock" : true
                            }, {
                              "relName" : "subsetPelagoUnpack#11595",
                              "attrName" : "s_nation",
                              "isBlock" : true
                            }, {
                              "relName" : "subsetPelagoUnpack#11595",
                              "attrName" : "p_category",
                              "isBlock" : true
                            }, {
                              "relName" : "subsetPelagoUnpack#11595",
                              "attrName" : "profit",
                              "isBlock" : true
                            } ],
                            "input" : {
                              "operator" : "gpu-to-cpu",
                              "projections" : [ {
                                "relName" : "subsetPelagoUnpack#11595",
                                "attrName" : "d_year",
                                "isBlock" : true
                              }, {
                                "relName" : "subsetPelagoUnpack#11595",
                                "attrName" : "s_nation",
                                "isBlock" : true
                              }, {
                                "relName" : "subsetPelagoUnpack#11595",
                                "attrName" : "p_category",
                                "isBlock" : true
                              }, {
                                "relName" : "subsetPelagoUnpack#11595",
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
                                      "relName" : "subsetPelagoUnpack#11595"
                                    } ],
                                    "type" : {
                                      "relName" : "subsetPelagoUnpack#11595",
                                      "type" : "record"
                                    },
                                    "argNo" : -1
                                  },
                                  "attribute" : {
                                    "attrName" : "d_year",
                                    "relName" : "subsetPelagoUnpack#11595"
                                  }
                                }, {
                                  "expression" : "recordProjection",
                                  "e" : {
                                    "expression" : "argument",
                                    "attributes" : [ {
                                      "attrName" : "s_nation",
                                      "relName" : "subsetPelagoUnpack#11595"
                                    } ],
                                    "type" : {
                                      "relName" : "subsetPelagoUnpack#11595",
                                      "type" : "record"
                                    },
                                    "argNo" : -1
                                  },
                                  "attribute" : {
                                    "attrName" : "s_nation",
                                    "relName" : "subsetPelagoUnpack#11595"
                                  }
                                }, {
                                  "expression" : "recordProjection",
                                  "e" : {
                                    "expression" : "argument",
                                    "attributes" : [ {
                                      "attrName" : "p_category",
                                      "relName" : "subsetPelagoUnpack#11595"
                                    } ],
                                    "type" : {
                                      "relName" : "subsetPelagoUnpack#11595",
                                      "type" : "record"
                                    },
                                    "argNo" : -1
                                  },
                                  "attribute" : {
                                    "attrName" : "p_category",
                                    "relName" : "subsetPelagoUnpack#11595"
                                  }
                                }, {
                                  "expression" : "recordProjection",
                                  "e" : {
                                    "expression" : "argument",
                                    "attributes" : [ {
                                      "attrName" : "profit",
                                      "relName" : "subsetPelagoUnpack#11595"
                                    } ],
                                    "type" : {
                                      "relName" : "subsetPelagoUnpack#11595",
                                      "type" : "record"
                                    },
                                    "argNo" : -1
                                  },
                                  "attribute" : {
                                    "attrName" : "profit",
                                    "relName" : "subsetPelagoUnpack#11595"
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
                                        "relName" : "subsetPelagoProject#11589"
                                      } ],
                                      "type" : {
                                        "relName" : "subsetPelagoProject#11589",
                                        "type" : "record"
                                      },
                                      "argNo" : -1
                                    },
                                    "attribute" : {
                                      "attrName" : "d_year",
                                      "relName" : "subsetPelagoProject#11589"
                                    },
                                    "register_as" : {
                                      "attrName" : "d_year",
                                      "relName" : "subsetPelagoUnpack#11595"
                                    }
                                  }, {
                                    "expression" : "recordProjection",
                                    "e" : {
                                      "expression" : "argument",
                                      "attributes" : [ {
                                        "attrName" : "s_nation",
                                        "relName" : "subsetPelagoProject#11589"
                                      } ],
                                      "type" : {
                                        "relName" : "subsetPelagoProject#11589",
                                        "type" : "record"
                                      },
                                      "argNo" : -1
                                    },
                                    "attribute" : {
                                      "attrName" : "s_nation",
                                      "relName" : "subsetPelagoProject#11589"
                                    },
                                    "register_as" : {
                                      "attrName" : "s_nation",
                                      "relName" : "subsetPelagoUnpack#11595"
                                    }
                                  }, {
                                    "expression" : "recordProjection",
                                    "e" : {
                                      "expression" : "argument",
                                      "attributes" : [ {
                                        "attrName" : "p_category",
                                        "relName" : "subsetPelagoProject#11589"
                                      } ],
                                      "type" : {
                                        "relName" : "subsetPelagoProject#11589",
                                        "type" : "record"
                                      },
                                      "argNo" : -1
                                    },
                                    "attribute" : {
                                      "attrName" : "p_category",
                                      "relName" : "subsetPelagoProject#11589"
                                    },
                                    "register_as" : {
                                      "attrName" : "p_category",
                                      "relName" : "subsetPelagoUnpack#11595"
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
                                          "relName" : "subsetPelagoProject#11589"
                                        } ],
                                        "type" : {
                                          "relName" : "subsetPelagoProject#11589",
                                          "type" : "record"
                                        },
                                        "argNo" : -1
                                      },
                                      "attribute" : {
                                        "attrName" : "$f3",
                                        "relName" : "subsetPelagoProject#11589"
                                      },
                                      "register_as" : {
                                        "attrName" : "profit",
                                        "relName" : "subsetPelagoUnpack#11595"
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
                                    "relName" : "subsetPelagoProject#11589",
                                    "e" : [ {
                                      "expression" : "recordProjection",
                                      "e" : {
                                        "expression" : "argument",
                                        "attributes" : [ {
                                          "attrName" : "d_year",
                                          "relName" : "subsetPelagoProject#11589"
                                        } ],
                                        "type" : {
                                          "relName" : "subsetPelagoProject#11589",
                                          "type" : "record"
                                        },
                                        "argNo" : -1
                                      },
                                      "attribute" : {
                                        "attrName" : "d_year",
                                        "relName" : "subsetPelagoProject#11589"
                                      },
                                      "register_as" : {
                                        "attrName" : "d_year",
                                        "relName" : "subsetPelagoProject#11589"
                                      }
                                    }, {
                                      "expression" : "recordProjection",
                                      "e" : {
                                        "expression" : "argument",
                                        "attributes" : [ {
                                          "attrName" : "s_nation",
                                          "relName" : "subsetPelagoProject#11589"
                                        } ],
                                        "type" : {
                                          "relName" : "subsetPelagoProject#11589",
                                          "type" : "record"
                                        },
                                        "argNo" : -1
                                      },
                                      "attribute" : {
                                        "attrName" : "s_nation",
                                        "relName" : "subsetPelagoProject#11589"
                                      },
                                      "register_as" : {
                                        "attrName" : "s_nation",
                                        "relName" : "subsetPelagoProject#11589"
                                      }
                                    }, {
                                      "expression" : "recordProjection",
                                      "e" : {
                                        "expression" : "argument",
                                        "attributes" : [ {
                                          "attrName" : "p_category",
                                          "relName" : "subsetPelagoProject#11589"
                                        } ],
                                        "type" : {
                                          "relName" : "subsetPelagoProject#11589",
                                          "type" : "record"
                                        },
                                        "argNo" : -1
                                      },
                                      "attribute" : {
                                        "attrName" : "p_category",
                                        "relName" : "subsetPelagoProject#11589"
                                      },
                                      "register_as" : {
                                        "attrName" : "p_category",
                                        "relName" : "subsetPelagoProject#11589"
                                      }
                                    }, {
                                      "expression" : "recordProjection",
                                      "e" : {
                                        "expression" : "argument",
                                        "attributes" : [ {
                                          "attrName" : "-",
                                          "relName" : "subsetPelagoProject#11589"
                                        } ],
                                        "type" : {
                                          "relName" : "subsetPelagoProject#11589",
                                          "type" : "record"
                                        },
                                        "argNo" : -1
                                      },
                                      "attribute" : {
                                        "attrName" : "-",
                                        "relName" : "subsetPelagoProject#11589"
                                      },
                                      "register_as" : {
                                        "attrName" : "$f3",
                                        "relName" : "subsetPelagoProject#11589"
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
                                            "relName" : "subsetPelagoProject#11563"
                                          } ],
                                          "type" : {
                                            "relName" : "subsetPelagoProject#11563",
                                            "type" : "record"
                                          },
                                          "argNo" : -1
                                        },
                                        "attribute" : {
                                          "attrName" : "c_custkey",
                                          "relName" : "subsetPelagoProject#11563"
                                        },
                                        "register_as" : {
                                          "attrName" : "$1",
                                          "relName" : "subsetPelagoProject#11589"
                                        }
                                      },
                                      "build_e" : [ {
                                        "e" : {
                                          "expression" : "recordProjection",
                                          "e" : {
                                            "expression" : "argument",
                                            "attributes" : [ {
                                              "attrName" : "c_custkey",
                                              "relName" : "subsetPelagoProject#11563"
                                            } ],
                                            "type" : {
                                              "relName" : "subsetPelagoProject#11563",
                                              "type" : "record"
                                            },
                                            "argNo" : -1
                                          },
                                          "attribute" : {
                                            "attrName" : "c_custkey",
                                            "relName" : "subsetPelagoProject#11563"
                                          },
                                          "register_as" : {
                                            "attrName" : "c_custkey",
                                            "relName" : "subsetPelagoProject#11589"
                                          }
                                        },
                                        "packet" : 1,
                                        "offset" : 0
                                      } ],
                                      "build_w" : [ 64, 32 ],
                                      "build_input" : {
                                        "operator" : "project",
                                        "gpu" : true,
                                        "relName" : "subsetPelagoProject#11563",
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
                                            "relName" : "subsetPelagoProject#11563"
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
                                            "relName" : "subsetPelagoProject#11587"
                                          } ],
                                          "type" : {
                                            "relName" : "subsetPelagoProject#11587",
                                            "type" : "record"
                                          },
                                          "argNo" : -1
                                        },
                                        "attribute" : {
                                          "attrName" : "lo_custkey",
                                          "relName" : "subsetPelagoProject#11587"
                                        },
                                        "register_as" : {
                                          "attrName" : "$0",
                                          "relName" : "subsetPelagoProject#11589"
                                        }
                                      },
                                      "probe_e" : [ {
                                        "e" : {
                                          "expression" : "recordProjection",
                                          "e" : {
                                            "expression" : "argument",
                                            "attributes" : [ {
                                              "attrName" : "lo_custkey",
                                              "relName" : "subsetPelagoProject#11587"
                                            } ],
                                            "type" : {
                                              "relName" : "subsetPelagoProject#11587",
                                              "type" : "record"
                                            },
                                            "argNo" : -1
                                          },
                                          "attribute" : {
                                            "attrName" : "lo_custkey",
                                            "relName" : "subsetPelagoProject#11587"
                                          },
                                          "register_as" : {
                                            "attrName" : "lo_custkey",
                                            "relName" : "subsetPelagoProject#11589"
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
                                              "relName" : "subsetPelagoProject#11587"
                                            } ],
                                            "type" : {
                                              "relName" : "subsetPelagoProject#11587",
                                              "type" : "record"
                                            },
                                            "argNo" : -1
                                          },
                                          "attribute" : {
                                            "attrName" : "d_year",
                                            "relName" : "subsetPelagoProject#11587"
                                          },
                                          "register_as" : {
                                            "attrName" : "d_year",
                                            "relName" : "subsetPelagoProject#11589"
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
                                              "attrName" : "s_nation",
                                              "relName" : "subsetPelagoProject#11587"
                                            } ],
                                            "type" : {
                                              "relName" : "subsetPelagoProject#11587",
                                              "type" : "record"
                                            },
                                            "argNo" : -1
                                          },
                                          "attribute" : {
                                            "attrName" : "s_nation",
                                            "relName" : "subsetPelagoProject#11587"
                                          },
                                          "register_as" : {
                                            "attrName" : "s_nation",
                                            "relName" : "subsetPelagoProject#11589"
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
                                              "attrName" : "p_category",
                                              "relName" : "subsetPelagoProject#11587"
                                            } ],
                                            "type" : {
                                              "relName" : "subsetPelagoProject#11587",
                                              "type" : "record"
                                            },
                                            "argNo" : -1
                                          },
                                          "attribute" : {
                                            "attrName" : "p_category",
                                            "relName" : "subsetPelagoProject#11587"
                                          },
                                          "register_as" : {
                                            "attrName" : "p_category",
                                            "relName" : "subsetPelagoProject#11589"
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
                                              "relName" : "subsetPelagoProject#11587"
                                            } ],
                                            "type" : {
                                              "relName" : "subsetPelagoProject#11587",
                                              "type" : "record"
                                            },
                                            "argNo" : -1
                                          },
                                          "attribute" : {
                                            "attrName" : "-",
                                            "relName" : "subsetPelagoProject#11587"
                                          },
                                          "register_as" : {
                                            "attrName" : "-",
                                            "relName" : "subsetPelagoProject#11589"
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
                                        "relName" : "subsetPelagoProject#11587",
                                        "e" : [ {
                                          "expression" : "recordProjection",
                                          "e" : {
                                            "expression" : "argument",
                                            "attributes" : [ {
                                              "attrName" : "lo_custkey",
                                              "relName" : "subsetPelagoProject#11587"
                                            } ],
                                            "type" : {
                                              "relName" : "subsetPelagoProject#11587",
                                              "type" : "record"
                                            },
                                            "argNo" : -1
                                          },
                                          "attribute" : {
                                            "attrName" : "lo_custkey",
                                            "relName" : "subsetPelagoProject#11587"
                                          },
                                          "register_as" : {
                                            "attrName" : "lo_custkey",
                                            "relName" : "subsetPelagoProject#11587"
                                          }
                                        }, {
                                          "expression" : "recordProjection",
                                          "e" : {
                                            "expression" : "argument",
                                            "attributes" : [ {
                                              "attrName" : "d_year",
                                              "relName" : "subsetPelagoProject#11587"
                                            } ],
                                            "type" : {
                                              "relName" : "subsetPelagoProject#11587",
                                              "type" : "record"
                                            },
                                            "argNo" : -1
                                          },
                                          "attribute" : {
                                            "attrName" : "d_year",
                                            "relName" : "subsetPelagoProject#11587"
                                          },
                                          "register_as" : {
                                            "attrName" : "d_year",
                                            "relName" : "subsetPelagoProject#11587"
                                          }
                                        }, {
                                          "expression" : "recordProjection",
                                          "e" : {
                                            "expression" : "argument",
                                            "attributes" : [ {
                                              "attrName" : "s_nation",
                                              "relName" : "subsetPelagoProject#11587"
                                            } ],
                                            "type" : {
                                              "relName" : "subsetPelagoProject#11587",
                                              "type" : "record"
                                            },
                                            "argNo" : -1
                                          },
                                          "attribute" : {
                                            "attrName" : "s_nation",
                                            "relName" : "subsetPelagoProject#11587"
                                          },
                                          "register_as" : {
                                            "attrName" : "s_nation",
                                            "relName" : "subsetPelagoProject#11587"
                                          }
                                        }, {
                                          "expression" : "recordProjection",
                                          "e" : {
                                            "expression" : "argument",
                                            "attributes" : [ {
                                              "attrName" : "p_category",
                                              "relName" : "subsetPelagoProject#11587"
                                            } ],
                                            "type" : {
                                              "relName" : "subsetPelagoProject#11587",
                                              "type" : "record"
                                            },
                                            "argNo" : -1
                                          },
                                          "attribute" : {
                                            "attrName" : "p_category",
                                            "relName" : "subsetPelagoProject#11587"
                                          },
                                          "register_as" : {
                                            "attrName" : "p_category",
                                            "relName" : "subsetPelagoProject#11587"
                                          }
                                        }, {
                                          "expression" : "recordProjection",
                                          "e" : {
                                            "expression" : "argument",
                                            "attributes" : [ {
                                              "attrName" : "-",
                                              "relName" : "subsetPelagoProject#11587"
                                            } ],
                                            "type" : {
                                              "relName" : "subsetPelagoProject#11587",
                                              "type" : "record"
                                            },
                                            "argNo" : -1
                                          },
                                          "attribute" : {
                                            "attrName" : "-",
                                            "relName" : "subsetPelagoProject#11587"
                                          },
                                          "register_as" : {
                                            "attrName" : "-",
                                            "relName" : "subsetPelagoProject#11587"
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
                                                "relName" : "subsetPelagoProject#11568"
                                              } ],
                                              "type" : {
                                                "relName" : "subsetPelagoProject#11568",
                                                "type" : "record"
                                              },
                                              "argNo" : -1
                                            },
                                            "attribute" : {
                                              "attrName" : "p_partkey",
                                              "relName" : "subsetPelagoProject#11568"
                                            },
                                            "register_as" : {
                                              "attrName" : "$3",
                                              "relName" : "subsetPelagoProject#11587"
                                            }
                                          },
                                          "build_e" : [ {
                                            "e" : {
                                              "expression" : "recordProjection",
                                              "e" : {
                                                "expression" : "argument",
                                                "attributes" : [ {
                                                  "attrName" : "p_partkey",
                                                  "relName" : "subsetPelagoProject#11568"
                                                } ],
                                                "type" : {
                                                  "relName" : "subsetPelagoProject#11568",
                                                  "type" : "record"
                                                },
                                                "argNo" : -1
                                              },
                                              "attribute" : {
                                                "attrName" : "p_partkey",
                                                "relName" : "subsetPelagoProject#11568"
                                              },
                                              "register_as" : {
                                                "attrName" : "p_partkey",
                                                "relName" : "subsetPelagoProject#11587"
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
                                                  "attrName" : "p_category",
                                                  "relName" : "subsetPelagoProject#11568"
                                                } ],
                                                "type" : {
                                                  "relName" : "subsetPelagoProject#11568",
                                                  "type" : "record"
                                                },
                                                "argNo" : -1
                                              },
                                              "attribute" : {
                                                "attrName" : "p_category",
                                                "relName" : "subsetPelagoProject#11568"
                                              },
                                              "register_as" : {
                                                "attrName" : "p_category",
                                                "relName" : "subsetPelagoProject#11587"
                                              }
                                            },
                                            "packet" : 2,
                                            "offset" : 0
                                          } ],
                                          "build_w" : [ 64, 32, 32 ],
                                          "build_input" : {
                                            "operator" : "project",
                                            "gpu" : true,
                                            "relName" : "subsetPelagoProject#11568",
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
                                                "relName" : "subsetPelagoProject#11568"
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
                                              },
                                              "register_as" : {
                                                "attrName" : "p_category",
                                                "relName" : "subsetPelagoProject#11568"
                                              }
                                            } ],
                                            "input" : {
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
                                                        "attrName" : "p_mfgr",
                                                        "relName" : "inputs/ssbm1000/part.csv"
                                                      } ],
                                                      "type" : {
                                                        "relName" : "inputs/ssbm1000/part.csv",
                                                        "type" : "record"
                                                      },
                                                      "argNo" : -1
                                                    },
                                                    "attribute" : {
                                                      "attrName" : "p_mfgr",
                                                      "relName" : "inputs/ssbm1000/part.csv"
                                                    }
                                                  },
                                                  "right" : {
                                                    "expression" : "dstring",
                                                    "v" : "MFGR#1",
                                                    "dict" : {
                                                      "path" : "inputs/ssbm1000/part.csv.ProjectedRelDataTypeField(#1: p_mfgr VARCHAR,null).dict"
                                                    }
                                                  }
                                                },
                                                "right" : {
                                                  "expression" : "eq",
                                                  "left" : {
                                                    "expression" : "recordProjection",
                                                    "e" : {
                                                      "expression" : "argument",
                                                      "attributes" : [ {
                                                        "attrName" : "p_mfgr",
                                                        "relName" : "inputs/ssbm1000/part.csv"
                                                      } ],
                                                      "type" : {
                                                        "relName" : "inputs/ssbm1000/part.csv",
                                                        "type" : "record"
                                                      },
                                                      "argNo" : -1
                                                    },
                                                    "attribute" : {
                                                      "attrName" : "p_mfgr",
                                                      "relName" : "inputs/ssbm1000/part.csv"
                                                    }
                                                  },
                                                  "right" : {
                                                    "expression" : "dstring",
                                                    "v" : "MFGR#2",
                                                    "dict" : {
                                                      "path" : "inputs/ssbm1000/part.csv.ProjectedRelDataTypeField(#1: p_mfgr VARCHAR,null).dict"
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
                                                      "attrName" : "p_mfgr",
                                                      "relName" : "inputs/ssbm1000/part.csv"
                                                    } ],
                                                    "type" : {
                                                      "relName" : "inputs/ssbm1000/part.csv",
                                                      "type" : "record"
                                                    },
                                                    "argNo" : -1
                                                  },
                                                  "attribute" : {
                                                    "attrName" : "p_mfgr",
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
                                                } ],
                                                "input" : {
                                                  "operator" : "cpu-to-gpu",
                                                  "projections" : [ {
                                                    "relName" : "inputs/ssbm1000/part.csv",
                                                    "attrName" : "p_partkey",
                                                    "isBlock" : true
                                                  }, {
                                                    "relName" : "inputs/ssbm1000/part.csv",
                                                    "attrName" : "p_mfgr",
                                                    "isBlock" : true
                                                  }, {
                                                    "relName" : "inputs/ssbm1000/part.csv",
                                                    "attrName" : "p_category",
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
                                                      "attrName" : "p_mfgr",
                                                      "isBlock" : true
                                                    }, {
                                                      "relName" : "inputs/ssbm1000/part.csv",
                                                      "attrName" : "p_category",
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
                                                        "attrName" : "p_mfgr",
                                                        "isBlock" : true
                                                      }, {
                                                        "relName" : "inputs/ssbm1000/part.csv",
                                                        "attrName" : "p_category",
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
                                                            "attrName" : "p_mfgr"
                                                          }, {
                                                            "relName" : "inputs/ssbm1000/part.csv",
                                                            "attrName" : "p_category"
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
                                                "relName" : "subsetPelagoProject#11585"
                                              } ],
                                              "type" : {
                                                "relName" : "subsetPelagoProject#11585",
                                                "type" : "record"
                                              },
                                              "argNo" : -1
                                            },
                                            "attribute" : {
                                              "attrName" : "lo_partkey",
                                              "relName" : "subsetPelagoProject#11585"
                                            },
                                            "register_as" : {
                                              "attrName" : "$0",
                                              "relName" : "subsetPelagoProject#11587"
                                            }
                                          },
                                          "probe_e" : [ {
                                            "e" : {
                                              "expression" : "recordProjection",
                                              "e" : {
                                                "expression" : "argument",
                                                "attributes" : [ {
                                                  "attrName" : "lo_custkey",
                                                  "relName" : "subsetPelagoProject#11585"
                                                } ],
                                                "type" : {
                                                  "relName" : "subsetPelagoProject#11585",
                                                  "type" : "record"
                                                },
                                                "argNo" : -1
                                              },
                                              "attribute" : {
                                                "attrName" : "lo_custkey",
                                                "relName" : "subsetPelagoProject#11585"
                                              },
                                              "register_as" : {
                                                "attrName" : "lo_custkey",
                                                "relName" : "subsetPelagoProject#11587"
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
                                                  "relName" : "subsetPelagoProject#11585"
                                                } ],
                                                "type" : {
                                                  "relName" : "subsetPelagoProject#11585",
                                                  "type" : "record"
                                                },
                                                "argNo" : -1
                                              },
                                              "attribute" : {
                                                "attrName" : "lo_partkey",
                                                "relName" : "subsetPelagoProject#11585"
                                              },
                                              "register_as" : {
                                                "attrName" : "lo_partkey",
                                                "relName" : "subsetPelagoProject#11587"
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
                                                  "relName" : "subsetPelagoProject#11585"
                                                } ],
                                                "type" : {
                                                  "relName" : "subsetPelagoProject#11585",
                                                  "type" : "record"
                                                },
                                                "argNo" : -1
                                              },
                                              "attribute" : {
                                                "attrName" : "d_year",
                                                "relName" : "subsetPelagoProject#11585"
                                              },
                                              "register_as" : {
                                                "attrName" : "d_year",
                                                "relName" : "subsetPelagoProject#11587"
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
                                                  "attrName" : "s_nation",
                                                  "relName" : "subsetPelagoProject#11585"
                                                } ],
                                                "type" : {
                                                  "relName" : "subsetPelagoProject#11585",
                                                  "type" : "record"
                                                },
                                                "argNo" : -1
                                              },
                                              "attribute" : {
                                                "attrName" : "s_nation",
                                                "relName" : "subsetPelagoProject#11585"
                                              },
                                              "register_as" : {
                                                "attrName" : "s_nation",
                                                "relName" : "subsetPelagoProject#11587"
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
                                                  "relName" : "subsetPelagoProject#11585"
                                                } ],
                                                "type" : {
                                                  "relName" : "subsetPelagoProject#11585",
                                                  "type" : "record"
                                                },
                                                "argNo" : -1
                                              },
                                              "attribute" : {
                                                "attrName" : "-",
                                                "relName" : "subsetPelagoProject#11585"
                                              },
                                              "register_as" : {
                                                "attrName" : "-",
                                                "relName" : "subsetPelagoProject#11587"
                                              }
                                            },
                                            "packet" : 5,
                                            "offset" : 0
                                          } ],
                                          "probe_w" : [ 64, 32, 32, 32, 32, 32 ],
                                          "hash_bits" : 21,
                                          "maxBuildInputSize" : 2000000,
                                          "probe_input" : {
                                            "operator" : "project",
                                            "gpu" : true,
                                            "relName" : "subsetPelagoProject#11585",
                                            "e" : [ {
                                              "expression" : "recordProjection",
                                              "e" : {
                                                "expression" : "argument",
                                                "attributes" : [ {
                                                  "attrName" : "lo_custkey",
                                                  "relName" : "subsetPelagoProject#11585"
                                                } ],
                                                "type" : {
                                                  "relName" : "subsetPelagoProject#11585",
                                                  "type" : "record"
                                                },
                                                "argNo" : -1
                                              },
                                              "attribute" : {
                                                "attrName" : "lo_custkey",
                                                "relName" : "subsetPelagoProject#11585"
                                              },
                                              "register_as" : {
                                                "attrName" : "lo_custkey",
                                                "relName" : "subsetPelagoProject#11585"
                                              }
                                            }, {
                                              "expression" : "recordProjection",
                                              "e" : {
                                                "expression" : "argument",
                                                "attributes" : [ {
                                                  "attrName" : "lo_partkey",
                                                  "relName" : "subsetPelagoProject#11585"
                                                } ],
                                                "type" : {
                                                  "relName" : "subsetPelagoProject#11585",
                                                  "type" : "record"
                                                },
                                                "argNo" : -1
                                              },
                                              "attribute" : {
                                                "attrName" : "lo_partkey",
                                                "relName" : "subsetPelagoProject#11585"
                                              },
                                              "register_as" : {
                                                "attrName" : "lo_partkey",
                                                "relName" : "subsetPelagoProject#11585"
                                              }
                                            }, {
                                              "expression" : "recordProjection",
                                              "e" : {
                                                "expression" : "argument",
                                                "attributes" : [ {
                                                  "attrName" : "d_year",
                                                  "relName" : "subsetPelagoProject#11585"
                                                } ],
                                                "type" : {
                                                  "relName" : "subsetPelagoProject#11585",
                                                  "type" : "record"
                                                },
                                                "argNo" : -1
                                              },
                                              "attribute" : {
                                                "attrName" : "d_year",
                                                "relName" : "subsetPelagoProject#11585"
                                              },
                                              "register_as" : {
                                                "attrName" : "d_year",
                                                "relName" : "subsetPelagoProject#11585"
                                              }
                                            }, {
                                              "expression" : "recordProjection",
                                              "e" : {
                                                "expression" : "argument",
                                                "attributes" : [ {
                                                  "attrName" : "s_nation",
                                                  "relName" : "subsetPelagoProject#11585"
                                                } ],
                                                "type" : {
                                                  "relName" : "subsetPelagoProject#11585",
                                                  "type" : "record"
                                                },
                                                "argNo" : -1
                                              },
                                              "attribute" : {
                                                "attrName" : "s_nation",
                                                "relName" : "subsetPelagoProject#11585"
                                              },
                                              "register_as" : {
                                                "attrName" : "s_nation",
                                                "relName" : "subsetPelagoProject#11585"
                                              }
                                            }, {
                                              "expression" : "recordProjection",
                                              "e" : {
                                                "expression" : "argument",
                                                "attributes" : [ {
                                                  "attrName" : "-",
                                                  "relName" : "subsetPelagoProject#11585"
                                                } ],
                                                "type" : {
                                                  "relName" : "subsetPelagoProject#11585",
                                                  "type" : "record"
                                                },
                                                "argNo" : -1
                                              },
                                              "attribute" : {
                                                "attrName" : "-",
                                                "relName" : "subsetPelagoProject#11585"
                                              },
                                              "register_as" : {
                                                "attrName" : "-",
                                                "relName" : "subsetPelagoProject#11585"
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
                                                    "relName" : "subsetPelagoProject#11573"
                                                  } ],
                                                  "type" : {
                                                    "relName" : "subsetPelagoProject#11573",
                                                    "type" : "record"
                                                  },
                                                  "argNo" : -1
                                                },
                                                "attribute" : {
                                                  "attrName" : "s_suppkey",
                                                  "relName" : "subsetPelagoProject#11573"
                                                },
                                                "register_as" : {
                                                  "attrName" : "$4",
                                                  "relName" : "subsetPelagoProject#11585"
                                                }
                                              },
                                              "build_e" : [ {
                                                "e" : {
                                                  "expression" : "recordProjection",
                                                  "e" : {
                                                    "expression" : "argument",
                                                    "attributes" : [ {
                                                      "attrName" : "s_suppkey",
                                                      "relName" : "subsetPelagoProject#11573"
                                                    } ],
                                                    "type" : {
                                                      "relName" : "subsetPelagoProject#11573",
                                                      "type" : "record"
                                                    },
                                                    "argNo" : -1
                                                  },
                                                  "attribute" : {
                                                    "attrName" : "s_suppkey",
                                                    "relName" : "subsetPelagoProject#11573"
                                                  },
                                                  "register_as" : {
                                                    "attrName" : "s_suppkey",
                                                    "relName" : "subsetPelagoProject#11585"
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
                                                      "attrName" : "s_nation",
                                                      "relName" : "subsetPelagoProject#11573"
                                                    } ],
                                                    "type" : {
                                                      "relName" : "subsetPelagoProject#11573",
                                                      "type" : "record"
                                                    },
                                                    "argNo" : -1
                                                  },
                                                  "attribute" : {
                                                    "attrName" : "s_nation",
                                                    "relName" : "subsetPelagoProject#11573"
                                                  },
                                                  "register_as" : {
                                                    "attrName" : "s_nation",
                                                    "relName" : "subsetPelagoProject#11585"
                                                  }
                                                },
                                                "packet" : 2,
                                                "offset" : 0
                                              } ],
                                              "build_w" : [ 64, 32, 32 ],
                                              "build_input" : {
                                                "operator" : "project",
                                                "gpu" : true,
                                                "relName" : "subsetPelagoProject#11573",
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
                                                    "relName" : "subsetPelagoProject#11573"
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
                                                  },
                                                  "register_as" : {
                                                    "attrName" : "s_nation",
                                                    "relName" : "subsetPelagoProject#11573"
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
                                                      "v" : "AMERICA",
                                                      "dict" : {
                                                        "path" : "inputs/ssbm1000/supplier.csv.ProjectedRelDataTypeField(#2: s_region VARCHAR,null).dict"
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
                                                        "attrName" : "s_nation",
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
                                                          "attrName" : "s_nation",
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
                                                            "attrName" : "s_nation",
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
                                                                "attrName" : "s_nation"
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
                                                    "relName" : "subsetPelagoProject#11583"
                                                  } ],
                                                  "type" : {
                                                    "relName" : "subsetPelagoProject#11583",
                                                    "type" : "record"
                                                  },
                                                  "argNo" : -1
                                                },
                                                "attribute" : {
                                                  "attrName" : "lo_suppkey",
                                                  "relName" : "subsetPelagoProject#11583"
                                                },
                                                "register_as" : {
                                                  "attrName" : "$0",
                                                  "relName" : "subsetPelagoProject#11585"
                                                }
                                              },
                                              "probe_e" : [ {
                                                "e" : {
                                                  "expression" : "recordProjection",
                                                  "e" : {
                                                    "expression" : "argument",
                                                    "attributes" : [ {
                                                      "attrName" : "lo_custkey",
                                                      "relName" : "subsetPelagoProject#11583"
                                                    } ],
                                                    "type" : {
                                                      "relName" : "subsetPelagoProject#11583",
                                                      "type" : "record"
                                                    },
                                                    "argNo" : -1
                                                  },
                                                  "attribute" : {
                                                    "attrName" : "lo_custkey",
                                                    "relName" : "subsetPelagoProject#11583"
                                                  },
                                                  "register_as" : {
                                                    "attrName" : "lo_custkey",
                                                    "relName" : "subsetPelagoProject#11585"
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
                                                      "relName" : "subsetPelagoProject#11583"
                                                    } ],
                                                    "type" : {
                                                      "relName" : "subsetPelagoProject#11583",
                                                      "type" : "record"
                                                    },
                                                    "argNo" : -1
                                                  },
                                                  "attribute" : {
                                                    "attrName" : "lo_partkey",
                                                    "relName" : "subsetPelagoProject#11583"
                                                  },
                                                  "register_as" : {
                                                    "attrName" : "lo_partkey",
                                                    "relName" : "subsetPelagoProject#11585"
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
                                                      "relName" : "subsetPelagoProject#11583"
                                                    } ],
                                                    "type" : {
                                                      "relName" : "subsetPelagoProject#11583",
                                                      "type" : "record"
                                                    },
                                                    "argNo" : -1
                                                  },
                                                  "attribute" : {
                                                    "attrName" : "lo_suppkey",
                                                    "relName" : "subsetPelagoProject#11583"
                                                  },
                                                  "register_as" : {
                                                    "attrName" : "lo_suppkey",
                                                    "relName" : "subsetPelagoProject#11585"
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
                                                      "relName" : "subsetPelagoProject#11583"
                                                    } ],
                                                    "type" : {
                                                      "relName" : "subsetPelagoProject#11583",
                                                      "type" : "record"
                                                    },
                                                    "argNo" : -1
                                                  },
                                                  "attribute" : {
                                                    "attrName" : "d_year",
                                                    "relName" : "subsetPelagoProject#11583"
                                                  },
                                                  "register_as" : {
                                                    "attrName" : "d_year",
                                                    "relName" : "subsetPelagoProject#11585"
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
                                                      "relName" : "subsetPelagoProject#11583"
                                                    } ],
                                                    "type" : {
                                                      "relName" : "subsetPelagoProject#11583",
                                                      "type" : "record"
                                                    },
                                                    "argNo" : -1
                                                  },
                                                  "attribute" : {
                                                    "attrName" : "-",
                                                    "relName" : "subsetPelagoProject#11583"
                                                  },
                                                  "register_as" : {
                                                    "attrName" : "-",
                                                    "relName" : "subsetPelagoProject#11585"
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
                                                "relName" : "subsetPelagoProject#11583",
                                                "e" : [ {
                                                  "expression" : "recordProjection",
                                                  "e" : {
                                                    "expression" : "argument",
                                                    "attributes" : [ {
                                                      "attrName" : "lo_custkey",
                                                      "relName" : "subsetPelagoProject#11583"
                                                    } ],
                                                    "type" : {
                                                      "relName" : "subsetPelagoProject#11583",
                                                      "type" : "record"
                                                    },
                                                    "argNo" : -1
                                                  },
                                                  "attribute" : {
                                                    "attrName" : "lo_custkey",
                                                    "relName" : "subsetPelagoProject#11583"
                                                  },
                                                  "register_as" : {
                                                    "attrName" : "lo_custkey",
                                                    "relName" : "subsetPelagoProject#11583"
                                                  }
                                                }, {
                                                  "expression" : "recordProjection",
                                                  "e" : {
                                                    "expression" : "argument",
                                                    "attributes" : [ {
                                                      "attrName" : "lo_partkey",
                                                      "relName" : "subsetPelagoProject#11583"
                                                    } ],
                                                    "type" : {
                                                      "relName" : "subsetPelagoProject#11583",
                                                      "type" : "record"
                                                    },
                                                    "argNo" : -1
                                                  },
                                                  "attribute" : {
                                                    "attrName" : "lo_partkey",
                                                    "relName" : "subsetPelagoProject#11583"
                                                  },
                                                  "register_as" : {
                                                    "attrName" : "lo_partkey",
                                                    "relName" : "subsetPelagoProject#11583"
                                                  }
                                                }, {
                                                  "expression" : "recordProjection",
                                                  "e" : {
                                                    "expression" : "argument",
                                                    "attributes" : [ {
                                                      "attrName" : "lo_suppkey",
                                                      "relName" : "subsetPelagoProject#11583"
                                                    } ],
                                                    "type" : {
                                                      "relName" : "subsetPelagoProject#11583",
                                                      "type" : "record"
                                                    },
                                                    "argNo" : -1
                                                  },
                                                  "attribute" : {
                                                    "attrName" : "lo_suppkey",
                                                    "relName" : "subsetPelagoProject#11583"
                                                  },
                                                  "register_as" : {
                                                    "attrName" : "lo_suppkey",
                                                    "relName" : "subsetPelagoProject#11583"
                                                  }
                                                }, {
                                                  "expression" : "recordProjection",
                                                  "e" : {
                                                    "expression" : "argument",
                                                    "attributes" : [ {
                                                      "attrName" : "d_year",
                                                      "relName" : "subsetPelagoProject#11583"
                                                    } ],
                                                    "type" : {
                                                      "relName" : "subsetPelagoProject#11583",
                                                      "type" : "record"
                                                    },
                                                    "argNo" : -1
                                                  },
                                                  "attribute" : {
                                                    "attrName" : "d_year",
                                                    "relName" : "subsetPelagoProject#11583"
                                                  },
                                                  "register_as" : {
                                                    "attrName" : "d_year",
                                                    "relName" : "subsetPelagoProject#11583"
                                                  }
                                                }, {
                                                  "expression" : "recordProjection",
                                                  "e" : {
                                                    "expression" : "argument",
                                                    "attributes" : [ {
                                                      "attrName" : "-",
                                                      "relName" : "subsetPelagoProject#11583"
                                                    } ],
                                                    "type" : {
                                                      "relName" : "subsetPelagoProject#11583",
                                                      "type" : "record"
                                                    },
                                                    "argNo" : -1
                                                  },
                                                  "attribute" : {
                                                    "attrName" : "-",
                                                    "relName" : "subsetPelagoProject#11583"
                                                  },
                                                  "register_as" : {
                                                    "attrName" : "-",
                                                    "relName" : "subsetPelagoProject#11583"
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
                                                      "relName" : "subsetPelagoProject#11583"
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
                                                        "relName" : "subsetPelagoProject#11583"
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
                                                        "relName" : "subsetPelagoProject#11583"
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
                                                        "relName" : "subsetPelagoProject#11581"
                                                      } ],
                                                      "type" : {
                                                        "relName" : "subsetPelagoProject#11581",
                                                        "type" : "record"
                                                      },
                                                      "argNo" : -1
                                                    },
                                                    "attribute" : {
                                                      "attrName" : "lo_orderdate",
                                                      "relName" : "subsetPelagoProject#11581"
                                                    },
                                                    "register_as" : {
                                                      "attrName" : "$0",
                                                      "relName" : "subsetPelagoProject#11583"
                                                    }
                                                  },
                                                  "probe_e" : [ {
                                                    "e" : {
                                                      "expression" : "recordProjection",
                                                      "e" : {
                                                        "expression" : "argument",
                                                        "attributes" : [ {
                                                          "attrName" : "lo_custkey",
                                                          "relName" : "subsetPelagoProject#11581"
                                                        } ],
                                                        "type" : {
                                                          "relName" : "subsetPelagoProject#11581",
                                                          "type" : "record"
                                                        },
                                                        "argNo" : -1
                                                      },
                                                      "attribute" : {
                                                        "attrName" : "lo_custkey",
                                                        "relName" : "subsetPelagoProject#11581"
                                                      },
                                                      "register_as" : {
                                                        "attrName" : "lo_custkey",
                                                        "relName" : "subsetPelagoProject#11583"
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
                                                          "relName" : "subsetPelagoProject#11581"
                                                        } ],
                                                        "type" : {
                                                          "relName" : "subsetPelagoProject#11581",
                                                          "type" : "record"
                                                        },
                                                        "argNo" : -1
                                                      },
                                                      "attribute" : {
                                                        "attrName" : "lo_partkey",
                                                        "relName" : "subsetPelagoProject#11581"
                                                      },
                                                      "register_as" : {
                                                        "attrName" : "lo_partkey",
                                                        "relName" : "subsetPelagoProject#11583"
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
                                                          "relName" : "subsetPelagoProject#11581"
                                                        } ],
                                                        "type" : {
                                                          "relName" : "subsetPelagoProject#11581",
                                                          "type" : "record"
                                                        },
                                                        "argNo" : -1
                                                      },
                                                      "attribute" : {
                                                        "attrName" : "lo_suppkey",
                                                        "relName" : "subsetPelagoProject#11581"
                                                      },
                                                      "register_as" : {
                                                        "attrName" : "lo_suppkey",
                                                        "relName" : "subsetPelagoProject#11583"
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
                                                          "relName" : "subsetPelagoProject#11581"
                                                        } ],
                                                        "type" : {
                                                          "relName" : "subsetPelagoProject#11581",
                                                          "type" : "record"
                                                        },
                                                        "argNo" : -1
                                                      },
                                                      "attribute" : {
                                                        "attrName" : "lo_orderdate",
                                                        "relName" : "subsetPelagoProject#11581"
                                                      },
                                                      "register_as" : {
                                                        "attrName" : "lo_orderdate",
                                                        "relName" : "subsetPelagoProject#11583"
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
                                                          "relName" : "subsetPelagoProject#11581"
                                                        } ],
                                                        "type" : {
                                                          "relName" : "subsetPelagoProject#11581",
                                                          "type" : "record"
                                                        },
                                                        "argNo" : -1
                                                      },
                                                      "attribute" : {
                                                        "attrName" : "-",
                                                        "relName" : "subsetPelagoProject#11581"
                                                      },
                                                      "register_as" : {
                                                        "attrName" : "-",
                                                        "relName" : "subsetPelagoProject#11583"
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
                                                    "relName" : "subsetPelagoProject#11581",
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
                                                        "relName" : "subsetPelagoProject#11581"
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
                                                        "relName" : "subsetPelagoProject#11581"
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
                                                        "relName" : "subsetPelagoProject#11581"
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
                                                        "relName" : "subsetPelagoProject#11581"
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
                                                        "relName" : "subsetPelagoProject#11581"
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